#!/usr/bin/env python3
"""patch_flags.py -- open the coupling for the artifact tools, and stash the real PIT index.

Edits (idempotent, exact anchors verified on the tree):

  model:
    * every `... SIDECAR_P3_BETA ... > 0` condition also fires on SIDECAR_KEEP_FEATS>0,
      so the stream and the clean features are STORED (graph-connected) without turning
      any loss on;
    * `last_loss_min_idx` is stashed right after the CE-PIT argmin, so the artifact tool
      can read the REAL assignment instead of reconstructing it.

  dataset:
    * `clean_audios` is built when SIDECAR_LOAD_CLEAN>0 as well, independent of the losses.

  p3_artifacts_r2.py:
    * cmd_grads runs with P3_BETA=0 + KEEP_FEATS=1 and uses `l_ce = out.loss` directly
      (no subtraction of two terms).

Usage: python patch_flags.py --root <tree>
"""
import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

MODEL = "model/modeling_whisper_sidecar.py"
DATA = "utils/dataset.py"
ART = "p3_artifacts_r2.py"

BETA_NEEDLE = 'float(__import__(\'os\').environ.get("SIDECAR_P3_BETA", "0") or 0) > 0'
BETA_WITH_FLAG = ('(float(__import__(\'os\').environ.get("SIDECAR_P3_BETA", "0") or 0) > 0 '
                  'or float(__import__(\'os\').environ.get("SIDECAR_KEEP_FEATS", "0") or 0) > 0)')

DATA_OLD = 'if _croot and float(os.environ.get("SIDECAR_REC_LAM", "0") or 0) > 0:'
DATA_NEW = ('if _croot and (float(os.environ.get("SIDECAR_REC_LAM", "0") or 0) > 0\n'
            '                         or float(os.environ.get("SIDECAR_LOAD_CLEAN", "0") or 0) > 0):')

PIT_OLD = ("            loss, loss_min_idx= loss.view(-1, self.num_perms, self.num_spks, "
           "targets.shape[-1]).mean(dim=-1).mean(2).min(1)\n")
PIT_NEW = PIT_OLD + ("            self.model.encoder.last_loss_min_idx = loss_min_idx.detach()  "
                     "  # P3: the real CE-PIT assignment\n")

ART_OLD = '    os.environ["SIDECAR_P3_BETA"] = str(a.beta)      # term ON so the stream is stored'
ART_NEW = ('    os.environ["SIDECAR_P3_BETA"] = "0"             # CE only: no auxiliary in out.loss\n'
           '    os.environ["SIDECAR_KEEP_FEATS"] = "1"         # features stored, graph kept')

ART_OLD2 = '        l_ce = l_tot - a.beta * l_aux                 # exact graph subtraction'
ART_NEW2 = '        l_ce = l_tot                                   # pure CE: P3_BETA=0'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def apply_file(path, edits, backup=True):
    p = Path(path)
    txt = p.read_text()
    before = txt
    report = []
    for old, new, tag in edits:
        n = txt.count(old)
        report.append((tag, n))
        if n:
            txt = txt.replace(old, new)
    if txt != before:
        if backup:
            shutil.copy2(p, str(p) + ".bak_flags")
        p.write_text(txt)
    return report, txt != before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    m = root / MODEL
    d = root / DATA
    art = root / ART

    if a.check:
        t = m.read_text()
        print("model: KEEP_FEATS gates =", t.count("SIDECAR_KEEP_FEATS"),
              "| last_loss_min_idx =", t.count("last_loss_min_idx.detach()"))
        print("dataset: LOAD_CLEAN =", d.read_text().count("SIDECAR_LOAD_CLEAN"))
        if art.exists():
            print("artifact: KEEP_FEATS =", art.read_text().count("SIDECAR_KEEP_FEATS"))
        return 0

    rep, changed = apply_file(m, [(BETA_NEEDLE, BETA_WITH_FLAG, "model:BETA->+KEEP_FEATS"),
                                  (PIT_OLD, PIT_NEW, "model:stash last_loss_min_idx")])
    for tag, n in rep:
        print("  %-34s replacements=%d" % (tag, n))
    if rep[0][1] != 3:
        print("!! expected 3 BETA gates, found", rep[0][1])
        return 1

    rep2, ch2 = apply_file(d, [(DATA_OLD, DATA_NEW, "dataset:LOAD_CLEAN")])
    for tag, n in rep2:
        print("  %-34s replacements=%d" % (tag, n))
    if rep2[0][1] != 1:
        print("!! expected 1 dataset gate, found", rep2[0][1])
        return 1

    if art.exists():
        rep3, _ = apply_file(art, [(ART_OLD, ART_NEW, "artifact:P3_BETA=0+KEEP_FEATS"),
                                   (ART_OLD2, ART_NEW2, "artifact:l_ce=out.loss")])
        for tag, n in rep3:
            print("  %-34s replacements=%d" % (tag, n))

    # the model must expose the stashed index to the artifact tool
    print("shas:", {str(p.relative_to(root)): sha(p) for p in (m, d) if p.exists()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
