#!/usr/bin/env python3
"""Add distance control to the prompt, so position can be varied without reordering.

RESULTS.md section 6 tried to separate modality from position by moving the graph
soft prompts in front of the SMILES span. Reordering alone collapses generation to
0.01 BLEU-2, so the instrument told us nothing about section 3's confound.

This patch takes the other route. It leaves the order untouched and varies the
*distance* between each channel and the generation point by inserting neutral
filler text:

    MOLCA_FILLER_MID=k   filler between the SMILES span and the soft prompts,
                         pushing the SMILES k units further from generation
                         while the graph stays exactly where it was
    MOLCA_FILLER_END=k   filler after the whole prompt, moving both channels
                         away equally, which is the control for "does inserting
                         k neutral tokens hurt at all"
    MOLCA_FILLER_TEXT    the unit, repeated k times. It defaults to a clause
                         true of every molecule, so it adds distance rather
                         than information

The edit is line-anchored rather than block-anchored, because the two machines
patched `smiles_handler` differently: the cluster added an early-return for
`MOLCA_MOL_FIRST` while the laptop nested `MOLCA_GRAPH_FIRST` inside the
`is_gal` branch. Anchoring on the substitution line itself works on both.

Idempotent. Run it from the MolCA working copy:

    python apply_filler_patch.py            # or --revert
"""
import argparse
import io
import os
import sys

TARGET = os.path.join("data_provider", "stage2_chebi_dm.py")
MARK = "MOLCA_FILLER_MID"

# the galactica-path substitution, the one line both variants share
ANCHOR = "text = CUSTOM_SEQ_RE.sub(r'\\1\\3\\4%s' % (mol_ph), text)"
ESCAPE = "text = escape_custom_split_sequence(text)"
DEFS = "_u = os.environ.get('MOLCA_FILLER_TEXT', 'It is a chemical entity. ')"


def patch(lines):
    out = []
    did_defs = did_anchor = did_escape = 0
    for line in lines:
        st = line.strip()
        pad = line[:len(line) - len(line.lstrip())]

        if st.startswith("def smiles_handler"):
            out.append(line)
            out.append("    import os")
            out.append("    " + DEFS)
            out.append("    _fm = int(os.environ.get('MOLCA_FILLER_MID', '0'))")
            out.append("    _fe = int(os.environ.get('MOLCA_FILLER_END', '0'))")
            did_defs += 1
            continue

        if st == ANCHOR and not did_anchor:
            out.append(pad + "if _fm:")
            out.append(pad + "    text = CUSTOM_SEQ_RE.sub("
                             "r'\\1\\3\\4' + _u * _fm + mol_ph, text)")
            out.append(pad + "else:")
            out.append(pad + "    " + ANCHOR)
            did_anchor += 1
            continue

        if st == ESCAPE and did_anchor and not did_escape:
            out.append(line)
            out.append(pad + "if _fe:")
            out.append(pad + "    text = text + _u * _fe")
            did_escape += 1
            continue

        out.append(line)
    return out, (did_defs, did_anchor, did_escape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit("run this from the MolCA working copy: %s not found" % args.file)

    src = io.open(args.file, encoding="utf-8").read()
    backup = args.file + ".prefiller"

    if args.revert:
        if not os.path.exists(backup):
            sys.exit("no %s to restore from" % backup)
        io.open(args.file, "w", encoding="utf-8", newline="\n").write(
            io.open(backup, encoding="utf-8").read())
        os.remove(backup)
        print("reverted", args.file)
        return 0

    if MARK in src:
        print("already patched")
        return 0

    lines = src.split("\n")
    out, counts = patch(lines)
    if counts != (1, 1, 1):
        sys.exit("expected one definition site, one substitution and one escape "
                 "call, found %s. The file has drifted; patch it by hand" % (counts,))

    io.open(backup, "w", encoding="utf-8", newline="\n").write(src)
    io.open(args.file, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print("patched %s, original saved to %s" % (args.file, backup))
    return 0


if __name__ == "__main__":
    sys.exit(main())
