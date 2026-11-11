#!/usr/bin/env python3
"""PATCH P3 -- wire the target-vs-interferer margin loss into the model.

Applies on top of LOSS_V3 (which already stores last_full_streams / last_clean_feats /
last_mix_feats and takes clean_audios).  P3 only ADDS a gated block, so with
SIDECAR_P3_BETA unset the run is bit-identical to v3.

What the block does (all shapes verified on the server with a real batch):

    stream   last_full_streams (B*S, C, T)  -> (B, S, T, C)   [grad to separator]
    clean    last_clean_feats  (B*S, T, C)  -> (B, S, T, C)   [detached]
    assign   assignment_from_pit(perms, loss_min_idx, 2)      [from ASR PIT]
    active   A_i & A_other, with A from frame_activity(clean_audios)  [B,S,T]
    centre   fixed artifact from SIDECAR_P3_CENTER            [C]
    margin   fixed scalar from SIDECAR_P3_MARGIN

    loss = CE_PIT + beta * p3_margin_loss(...)

Everything (centre, mask, assignment, NaN handling) lives in p3_loss.py, which is
copied into the code tree so the model can import it.

Usage
    cp p3_loss.py fixed_retrain/p3_loss.py
    python patch_p3.py --apply  --root ~/ws/fixed_retrain
    python patch_p3.py --check  --root ~/ws/fixed_retrain
    python patch_p3.py --revert --root ~/ws/fixed_retrain
"""
import argparse
import shutil
import sys
from pathlib import Path

M = "P3"
MODEL = "model/modeling_whisper_sidecar.py"
HELPER = "p3_loss.py"

BLOCK = """            # ---- P3: target-vs-interferer margin on the stream ----------------
            _p3_beta = float(__import__('os').environ.get("SIDECAR_P3_BETA", "0") or 0)
            if _p3_beta > 0 and self.training:
                from p3_loss import (p3_margin_loss, assignment_from_pit, frame_activity)
                _p3_margin = float(__import__('os').environ.get("SIDECAR_P3_MARGIN", "0.02"))
                if not hasattr(self, "_p3_center"):
                    _cp = __import__('os').environ.get("SIDECAR_P3_CENTER", "")
                    assert _cp and __import__('os').path.isfile(_cp), (
                        "P3: SIDECAR_P3_CENTER must point at the saved centre, got %r" % (_cp,))
                    self._p3_center = torch.load(_cp, map_location="cpu").float().reshape(-1)
                    print("P3: centre loaded from %s (dim %d)" % (_cp, self._p3_center.numel()))
                _st = getattr(self.model.encoder, "last_full_streams", None)
                _cl = getattr(self.model.encoder, "last_clean_feats", None)
                assert _st is not None and _cl is not None, (
                    "P3: beta>0 but the stream/clean tensors are missing")
                _B, _S = _st.shape[0] // self.num_spks, self.num_spks
                _T = _st.shape[-1]
                _stream = _st.float().view(_B, _S, _st.shape[1], _T).permute(0, 1, 3, 2)   # (B,S,T,C)
                _clean = _cl.float().view(_B, _S, _cl.shape[1], _cl.shape[2])              # (B,S,T,C)
                _asg = assignment_from_pit(list(itertools.permutations(range(_S))), loss_min_idx, _S)
                assert clean_audios is not None, "P3: needs clean_audios in the batch"
                _act = frame_activity(clean_audios.view(-1, clean_audios.shape[-1]), n_frames=_T)
                _act = _act.view(_B, _S, _T)                                              # (B,S,T)
                _act = _act & _act.flip(1)                                                # both speak
                _p3_loss, _p3_stats = p3_margin_loss(
                    _stream, _clean, _asg, _act, self._p3_center, _p3_margin)
                loss = loss + _p3_beta * _p3_loss
                if int(__import__('os').environ.get("SIDECAR_P3_VERBOSE", "0")):
                    print("P3 stats:", _p3_stats)
                del _stream, _clean, _act
            # ---- end P3 ---------------------------------------------------------
"""

STORE_EDITS = [
    ('            if float(__import__(\'os\').environ.get("SIDECAR_SEP_LAM", "0") or 0) > 0:\n'
     '                self.last_full_streams = _full.transpose(1, 2)',
     '            if (float(__import__(\'os\').environ.get("SIDECAR_SEP_LAM", "0") or 0) > 0\n'
     '                    or float(__import__(\'os\').environ.get("SIDECAR_P3_BETA", "0") or 0) > 0):\n'
     '                self.last_full_streams = _full.transpose(1, 2)'),
    ('        _need_rec = float(__import__(\'os\').environ.get("SIDECAR_REC_LAM", "0") or 0) > 0',
     '        _need_rec = (float(__import__(\'os\').environ.get("SIDECAR_REC_LAM", "0") or 0) > 0\n'
     '                     or float(__import__(\'os\').environ.get("SIDECAR_P3_BETA", "0") or 0) > 0)'),
    ('        if clean_audios is not None and _rec_lam > 0:',
     '        if clean_audios is not None and (_rec_lam > 0 or\n'
     '                                         float(__import__(\'os\').environ.get("SIDECAR_P3_BETA", "0") or 0) > 0):'),
]

ANCHOR = "            loss = loss.mean()\n"


def find_root(explicit=None):
    if explicit:
        return Path(explicit).resolve()
    here = Path(__file__).resolve().parent
    for c in (here, here / "fixed_retrain", Path.cwd(), Path.cwd() / "fixed_retrain"):
        if (c / MODEL).is_file():
            return c
    raise SystemExit("cannot locate %s" % MODEL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = find_root(a.root)
    f = root / MODEL
    txt = f.read_text()

    if a.check:
        print("model patched:", "# ---- P3:" in txt)
        helper = root / HELPER
        print("helper present:", helper.exists())
        return 0

    if a.revert:
        bak = f.with_suffix(f.suffix + ".bak_p3")
        if bak.exists():
            shutil.copy2(bak, f)
            print("reverted", f)
        else:
            print("no backup")
        return 0

    if not a.apply:
        ap.print_help()
        return 0

    if "# ---- P3:" in txt:
        print("already patched")
        return 0
    if "LOSS_V3" not in txt:
        print("!! LOSS_V3 must be applied first (P3 relies on its stored tensors)")
        return 1
    if txt.count(ANCHOR) != 1:
        print("!! loss anchor count =", txt.count(ANCHOR))
        return 1

    helper_src = Path(__file__).resolve().parent / HELPER
    if helper_src.exists() and not (root / HELPER).exists():
        shutil.copy2(helper_src, root / HELPER)
        print("copied helper ->", root / HELPER)

    # P3 must be self-sufficient: make the v3 stores also fire when P3 is on
    for old, new in STORE_EDITS:
        if txt.count(old) != 1:
            print("!! store anchor count = %d for %r" % (txt.count(old), old[:60]))
            return 1
        txt = txt.replace(old, new, 1)

    bak = f.with_suffix(f.suffix + ".bak_p3")
    shutil.copy2(f, bak)
    txt = txt.replace(ANCHOR, ANCHOR + BLOCK, 1)
    f.write_text(txt)
    print("patched", f)
    print("verify:", "# ---- P3:" in f.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

