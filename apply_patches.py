#!/usr/bin/env python3
"""apply_patches.py - MolCA reproduction harness.

Akshay Ashok (7071170), MLLP seminar 2026.

Idempotent: run it repeatedly, already-applied patches are skipped.

GROUP A - portability. The released code assumes the authors' training rig
(2x A100, bf16, an initialised torch.distributed process group). None of those
hold on a single consumer GPU, and each assumption fails in a different file.

GROUP B - ablation harness. Adds inference-time interventions on the two input
channels (SMILES text / 2D graph) so a released checkpoint can be probed
without retraining. Conditions are selected by environment variable.
"""
import io
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MolCA")

CONDITIONS = {
    "MOLCA_GRAPH_ONLY": "withhold the SMILES string, keep the graph soft prompts",
    "MOLCA_SHUFFLE_GRAPH": "give each molecule the NEXT molecule's graph (rotate +1)",
    "MOLCA_SHUFFLE_GRAPH_REV": "rotate -1 instead of +1 (replication control)",
    "MOLCA_SHUFFLE_SMILES": "rotate the SMILES instead of the graph (mirror control)",
    "MOLCA_REWIRE_GRAPH": "keep atoms, resample edge_index uniformly (destroy topology)",
    "MOLCA_NULL_GRAPH": "zero the atom features (uninformative graph control)",
    "MOLCA_GRAPH_FIRST": "put the graph soft prompts BEFORE the SMILES span",
}


def edit(relpath, subs, label):
    """subs: list of (old, new). Replaces every occurrence of each `old`.

    Idempotency comes from `old` no longer being present after a successful
    pass, so re-running is a no-op. Do not write a `new` that still contains
    its own `old` - that makes the patch non-idempotent and, if two subs
    overlap, lets the second one rewrite the first one's output.
    """
    path = os.path.join(ROOT, relpath)
    text = io.open(path, encoding="utf-8").read()
    applied = 0
    for old, new in subs:
        assert old not in new, "%s: `old` appears inside `new`, patch is unsafe" % label
        n = text.count(old)
        if n:
            text = text.replace(old, new)
            applied += n
    if applied:
        io.open(path, "w", encoding="utf-8").write(text)
        print("  [+] %s (%d edit(s) in %s)" % (label, applied, relpath))
    else:
        print("  [=] %s (already applied / nothing to do)" % label)
    return True


def insert_before(relpath, anchor, block, sentinel, label):
    """Insert `block` immediately above `anchor`. No-op if `sentinel` is present.

    Used for adding new top-level definitions, where a plain (old -> new)
    substitution would necessarily contain its own anchor and so be unsafe.
    """
    path = os.path.join(ROOT, relpath)
    text = io.open(path, encoding="utf-8").read()
    if sentinel in text:
        print("  [=] %s (already applied)" % label)
        return True
    if anchor not in text:
        print("  !! %s: anchor not found in %s" % (label, relpath))
        return False
    io.open(path, "w", encoding="utf-8").write(
        text.replace(anchor, block.strip("\n") + "\n\n\n" + anchor, 1))
    print("  [+] %s (inserted into %s)" % (label, relpath))
    return True


ok = True
print("GROUP A - portability")

# A1. Checkpoints were saved on a multi-GPU node, so tensors carry cuda:4 device
#     tags. Unpickling on a 1-GPU machine raises. Force a CPU load and let
#     Lightning move the module to the device afterwards.
ok &= edit("stage2.py", [(
    "model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)",
    "model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args, "
    "map_location='cpu')",
)], "A1 checkpoint device tags (stage2)")

_stage1 = os.path.join(ROOT, "stage1.py")
if os.path.exists(_stage1):
    _s1 = io.open(_stage1, encoding="utf-8").read()
    _m = re.search(r"Blip2Stage1\.load_from_checkpoint\(([^)]*)\)", _s1)
    if _m and "map_location" not in _m.group(1):
        edit("stage1.py", [(_m.group(0), _m.group(0)[:-1] + ", map_location='cpu')")],
             "A1b checkpoint device tags (stage1)")

# A2. Galactica is hardcoded to bf16 in 2 of 3 branches. Cards without bf16 then
#     hit a dtype mismatch against the fp16 Q-Former output. Line 167 already
#     uses float16, so the code half-anticipates this case.
ok &= edit("model/blip2_opt.py", [(
    "OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)",
    "OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)",
)], "A2 hardcoded bf16 -> fp16")

# A3. The eval epoch-end hook gathers predictions across ranks unconditionally.
#     A single-process run has no process group, so this raises AFTER inference
#     has already finished - it discards a completed 2-hour run. Route every
#     call through one helper rather than nesting guards at each call site.
_HELPER = '''

def _gather_or_local(dst, src):
    """dist.all_gather_object, or its single-process equivalent.

    The released code calls all_gather_object unconditionally in the eval
    epoch-end hook. Without an initialised process group that raises, so a
    completed single-GPU inference pass is thrown away at the last step.
    """
    if dist.is_initialized():
        dist.all_gather_object(dst, src)
        return dst
    return [src]

'''
ok &= insert_before("model/blip2_stage2.py",
                    "def load_ignore_unexpected(model, state_dict):",
                    _HELPER, "_gather_or_local", "A3a _gather_or_local helper")

ok &= edit("model/blip2_stage2.py", [
    ("dist.all_gather_object(all_predictions, predictions)",
     "all_predictions = _gather_or_local(all_predictions, predictions)"),
    ("dist.all_gather_object(all_targets, targets)",
     "all_targets = _gather_or_local(all_targets, targets)"),
], "A3b route gathers through helper")

# A4. Same root cause inside the stage-1 contrastive loss.
ok &= edit("model/blip2qformer.py", [(
    "        rank = dist.get_rank()",
    "        rank = (dist.get_rank() if dist.is_initialized() else 0)",
)], "A4 unguarded dist.get_rank")

# A5. Every DataLoader hardcodes persistent_workers=True, which torch rejects
#     when num_workers=0. Single-process / Windows runs cannot use worker
#     processes freely, so num_workers=0 has to be a legal setting.
for _dm in ("data_provider/stage2_chebi_dm.py", "data_provider/stage2_dm.py",
            "data_provider/stage1_dm.py", "data_provider/iupac_dm.py"):
    if os.path.exists(os.path.join(ROOT, _dm)):
        edit(_dm, [("persistent_workers=True", "persistent_workers=(self.num_workers > 0)")],
             "A5 persistent_workers vs num_workers=0 [%s]" % os.path.basename(_dm))

print("")
print("GROUP B - ablation harness")

# B1. graph_only exists in stage2_dm.py (PubChem324k) but was never ported to
#     stage2_chebi_dm.py. Port it so the two datamodules expose the same option.
ok &= edit("data_provider/stage2_chebi_dm.py", [(
    "def smiles_handler(text, mol_ph, is_gal=True):\n"
    "    smiles_list = []\n"
    "    for match in CUSTOM_SEQ_RE.finditer(text):\n"
    "        smiles = match.group(3)\n"
    "        smiles_list.append(smiles)\n"
    "    if is_gal:",
    "def smiles_handler(text, mol_ph, is_gal=True):\n"
    "    smiles_list = []\n"
    "    for match in CUSTOM_SEQ_RE.finditer(text):\n"
    "        smiles = match.group(3)\n"
    "        smiles_list.append(smiles)\n"
    "    # ported from stage2_dm.py, which has a graph_only path this module lacked\n"
    "    if os.environ.get('MOLCA_GRAPH_ONLY') == '1':\n"
    "        return CUSTOM_SEQ_RE.sub(r'%s' % (mol_ph), text), smiles_list\n"
    "    if is_gal:",
)], "B1 graph_only (withhold SMILES)")

# B4. The template appends the graph soft prompts AFTER the SMILES span, so
#     they sit closer to the generation point than the text does. Any finding
#     that "the graph channel dominates" is therefore confounded with "the
#     nearest channel dominates". This flag puts the soft prompts in front so
#     modality and position can be separated.
ok &= edit("data_provider/stage2_chebi_dm.py", [(
    "    if is_gal:\n"
    "        text = CUSTOM_SEQ_RE.sub(r'\\1\\3\\4%s' % (mol_ph), text)",
    "    if is_gal:\n"
    "        if os.environ.get('MOLCA_GRAPH_FIRST') == '1':\n"
    "            text = CUSTOM_SEQ_RE.sub(r'%s\\1\\3\\4' % (mol_ph), text)\n"
    "        else:\n"
    "            text = CUSTOM_SEQ_RE.sub(r'\\1\\3\\4%s' % (mol_ph), text)",
)], "B4 graph-first prompt order (position vs modality)")

# B2/B3. Channel interventions, applied inside InferenceCollater only, so the
#        training collater is untouched. Each is a no-op unless its env var is 1.
_ABLATE = (
    "def _ablate(graphs, smiles_prompt):\n"
    '    """Inference-time channel interventions. No-op unless an env var is set.\n'
    "\n"
    "    Rotation is within-batch, so a substituted graph is always a real molecule\n"
    "    from the same split. That isolates 'wrong molecule' from 'invalid input'.\n"
    '    """\n'
    "    import copy as _copy\n"
    "    import torch as _torch\n"
    "    if os.environ.get('MOLCA_SHUFFLE_GRAPH') == '1':\n"
    "        graphs = graphs[1:] + graphs[:1]\n"
    "    if os.environ.get('MOLCA_SHUFFLE_GRAPH_REV') == '1':\n"
    "        graphs = graphs[-1:] + graphs[:-1]\n"
    "    if os.environ.get('MOLCA_SHUFFLE_SMILES') == '1':\n"
    "        smiles_prompt = smiles_prompt[1:] + smiles_prompt[:1]\n"
    "    if os.environ.get('MOLCA_REWIRE_GRAPH') == '1':\n"
    "        out = []\n"
    "        for g in graphs:\n"
    "            g = _copy.copy(g)\n"
    "            if g.edge_index is not None and g.edge_index.numel() > 0:\n"
    "                n, e = int(g.x.size(0)), g.edge_index.size(1)\n"
    "                g.edge_index = _torch.randint(0, n, (2, e), dtype=g.edge_index.dtype)\n"
    "            out.append(g)\n"
    "        graphs = tuple(out)\n"
    "    if os.environ.get('MOLCA_NULL_GRAPH') == '1':\n"
    "        out = []\n"
    "        for g in graphs:\n"
    "            g = _copy.copy(g)\n"
    "            g.x = _torch.zeros_like(g.x)\n"
    "            out.append(g)\n"
    "        graphs = tuple(out)\n"
    "    return graphs, smiles_prompt\n"
)
ok &= insert_before("data_provider/stage2_chebi_dm.py", "class InferenceCollater:",
                    _ABLATE, "def _ablate(", "B2 _ablate() intervention registry")

ok &= edit("data_provider/stage2_chebi_dm.py", [(
    "    def __call__(self, batch):\n"
    "        graphs, texts, smiles_prompt = zip(*batch)\n"
    "        graphs = self.collater(graphs)\n"
    "        smiles_prompt = [smiles_handler(p, self.mol_ph, self.is_gal)[0] for p in smiles_prompt]\n"
    "        ## deal with prompt\n"
    "        self.tokenizer.paddding_side = 'left'",
    "    def __call__(self, batch):\n"
    "        graphs, texts, smiles_prompt = zip(*batch)\n"
    "        graphs, smiles_prompt = _ablate(graphs, smiles_prompt)\n"
    "        graphs = self.collater(graphs)\n"
    "        smiles_prompt = [smiles_handler(p, self.mol_ph, self.is_gal)[0] for p in smiles_prompt]\n"
    "        ## deal with prompt\n"
    "        self.tokenizer.paddding_side = 'left'",
)], "B3 hook _ablate into InferenceCollater")

print("")
print("GROUP D - retrieval path (stage 1)")

# D1. stage1.py routes on the root path:
#         if args.root.find('kv') >= 0: Stage1KVPLMDM(... no tokenizer ...)
#         else:                         Stage1DM(... tokenizer ...)
#     The README's retrieval commands use --root data/kv_data, so they take the
#     first branch. Stage1KVPLMDM contains no reference to a tokenizer at all,
#     yet builds GINPretrainDataset, whose tokenizer_text() calls
#     self.tokenizer(...). trainer.validate() iterates exactly that dataset, so
#     every documented retrieval command dies with `NoneType is not callable`.
#     RetrievalDatasetKVPLM in the same codebase builds its own scibert
#     tokenizer; do the same here rather than change the constructor signature.
ok &= insert_before(
    "data_provider/stage1_kvplm_dm.py",
    "        self.val_dataset_match = RetrievalDatasetKVPLM(root + '/valid/', args).shuffle()",
    "        # Stage1KVPLMDM never receives a tokenizer, but GINPretrainDataset\n"
    "        # requires one. Mirror RetrievalDatasetKVPLM, which builds its own.\n"
    "        from transformers import BertTokenizer as _BertTokenizer\n"
    "        _tok = _BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')\n"
    "        self.train_dataset.tokenizer = _tok\n"
    "        self.val_dataset.tokenizer = _tok",
    "_tok", "D1 missing tokenizer on the KV-PLM retrieval path")

# D2. Same persistent_workers problem as A5; this file was not in that list.
edit("data_provider/stage1_kvplm_dm.py",
     [("persistent_workers=True", "persistent_workers=(self.num_workers > 0)")],
     "D2 persistent_workers vs num_workers=0 [stage1_kvplm_dm.py]")

# D3. torch 2.6 flipped torch.load's `weights_only` default to True. The graph
#     datasets store pickled PyG objects, so loading them now raises
#     "GLOBAL torch_geometric.data.storage.GlobalStorage was not an allowed
#     global". This cannot happen on the authors' torch 2.0, and did not show
#     up in captioning either, because ChEBI-20 ships .txt files while kv_data
#     ships pre-pickled .pt graphs.
#     weights_only=False re-enables arbitrary unpickling. That is acceptable
#     here and only here: these files come from the paper's own dataset.zip,
#     already on disk and already trusted. Do not carry this to untrusted data.
_LOADS = [
    ("torch.load(self.processed_paths[0])",
     "torch.load(self.processed_paths[0], weights_only=False)"),
    ("torch.load(graph_path)", "torch.load(graph_path, weights_only=False)"),
    ("torch.load(path)", "torch.load(path, weights_only=False)"),
]
for _f in ("data_provider/pretrain_dataset.py", "data_provider/retrieval_dataset.py",
           "data_provider/molecule_caption_dataset.py",
           "data_provider/molecule_iupac_dataset.py", "data_provider/loader.py"):
    if os.path.exists(os.path.join(ROOT, _f)):
        edit(_f, _LOADS, "D3 torch>=2.6 weights_only [%s]" % os.path.basename(_f))

print("")
print("GROUP C - harness convenience")

# C1. No way to run a short eval. A full CheBI-20 pass is ~3300 molecules with
#     beam search, so smoke-testing a config otherwise costs a full run.
ok &= edit("stage2.py", [(
    "trainer = Trainer(accelerator=args.accelerator, devices=args.devices, "
    "precision=args.precision, max_epochs=args.max_epochs, "
    "check_val_every_n_epoch=args.check_val_every_n_epoch, callbacks=callbacks, "
    "strategy=strategy, logger=logger)",
    "trainer = Trainer(accelerator=args.accelerator, devices=args.devices, "
    "precision=args.precision, max_epochs=args.max_epochs, "
    "check_val_every_n_epoch=args.check_val_every_n_epoch, callbacks=callbacks, "
    "strategy=strategy, logger=logger, limit_val_batches=args.limit_val_batches)",
)], "C1 --limit_val_batches (Trainer)")

ok &= insert_before(
    "stage2.py",
    "    parser.add_argument('--max_epochs', type=int, default=10)",
    "    # 1.0 = the whole split; a float <1 or an int N runs a subset (smoke tests)\n"
    "    parser.add_argument('--limit_val_batches', type=float, default=1.0)",
    "--limit_val_batches", "C1a --limit_val_batches (argparse)")

# C2. On Windows (WDDM) the driver lets a process oversubscribe VRAM into
#     system RAM instead of raising OOM. A batch that does not fit therefore
#     does not fail - it thrashes, collapses to ~0.03 it/s, exhausts system
#     RAM and hangs the machine. Capping the fraction converts that silent
#     death spiral into an immediate, diagnosable OOM.
ok &= insert_before(
    "stage2.py",
    "def main(args):",
    "def _cap_gpu_memory():\n"
    "    frac = os.environ.get('MOLCA_MEM_FRAC')\n"
    "    if frac and torch.cuda.is_available():\n"
    "        torch.cuda.set_per_process_memory_fraction(float(frac))\n"
    "        print('capped GPU memory fraction at %s' % frac)\n",
    "_cap_gpu_memory", "C2 GPU memory cap (avoids WDDM thrash-to-freeze)")

ok &= edit("stage2.py", [(
    "def main(args):\n    pl.seed_everything(args.seed)",
    "def main(args):\n    _cap_gpu_memory()\n    pl.seed_everything(args.seed)",
)], "C2a call the memory cap")

_s2 = os.path.join(ROOT, "stage2.py")
_t2 = io.open(_s2, encoding="utf-8").read()
if "args.limit_val_batches = int(args.limit_val_batches)" not in _t2:
    _a = "    trainer = Trainer(accelerator=args.accelerator"
    if _a in _t2:
        io.open(_s2, "w", encoding="utf-8").write(_t2.replace(
            _a,
            "    # Lightning wants an int for a batch count, a float for a fraction\n"
            "    if args.limit_val_batches > 1:\n"
            "        args.limit_val_batches = int(args.limit_val_batches)\n"
            + _a, 1))
        print("  [+] C1b limit_val_batches int/float coercion")
else:
    print("  [=] C1b limit_val_batches int/float coercion (already applied)")

# stage2_chebi_dm.py needs `os` for the env-var switches.
_chebi = os.path.join(ROOT, "data_provider/stage2_chebi_dm.py")
_txt = io.open(_chebi, encoding="utf-8").read()
if not re.search(r"^import os$", _txt, re.M):
    io.open(_chebi, "w", encoding="utf-8").write("import os\n" + _txt)
    print("  [+] added missing `import os`")

print("")
print("Active conditions this run:")
_active = [k for k in CONDITIONS if os.environ.get(k) == "1"]
print("  " + (", ".join(_active) if _active else "(none - baseline)"))
sys.exit(0 if ok else 1)
