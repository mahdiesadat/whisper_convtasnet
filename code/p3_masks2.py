#!/usr/bin/env python3
"""p3_masks2.py -- corrected masks + the collator that carries the valid lengths.

Three corrections from the review
  1. the valid length is captured BEFORE padding (in the collator, from the post-
     preprocessing arrays) and carried in the batch; `len(padded)` is never used and the
     length is never recovered from trailing zeros;
  2. `build_masks` takes BOTH the waveforms and the lengths, and the RMS/peak of the
     relative threshold are computed ONLY inside the valid region -- audio outside the
     crop/validity must not change the activity threshold inside it;
  3. the invariants assert with the right broadcasting (source_active (B,S,T) vs
     valid_mix (B,T) -> valid_mix[:, None, :]).

Wiring
    coll = LengthCollator(base_collator)          # returns batch + mix_length + clean_lengths
    batch = coll(items)
    masks = masks_from_batch(batch)               # union for the centre, overlap for P3
    centre uses masks["union_active"]; P3 uses masks["overlap_active"]
"""
from __future__ import annotations

import numpy as np
import torch

FRAME = 320          # 20 ms @ 16 kHz == one Whisper-small feature frame
N_FRAMES = 1500      # 30 s


# --------------------------------------------------------------------- masks --
def build_masks(src_wavs, src_lengths, mix_length, n_frames=N_FRAMES, frame=FRAME,
                rel_threshold=0.1, abs_floor=1e-3):
    """Four masks from waveforms + TRUE lengths.

    src_wavs     : (S, Lp) array-like, possibly right-padded
    src_lengths  : (S,) valid samples per source (pre-padding)
    mix_length   : scalar, valid samples of the mixture (pre-padding)
    returns dict with valid_mix (T,), valid_source (S,T), source_active (S,T),
                  union_active (T,), overlap_active (T,)
    """
    S = len(src_wavs)
    mix_blocks = min(int(mix_length) // frame, n_frames)
    valid_mix = torch.zeros(n_frames, dtype=torch.bool)
    valid_mix[:mix_blocks] = True

    valid_source = torch.zeros(S, n_frames, dtype=torch.bool)
    raw_active = torch.zeros(S, n_frames, dtype=torch.bool)
    for i in range(S):
        nb = min(int(src_lengths[i]) // frame, n_frames, mix_blocks)   # never past valid_mix
        valid_source[i, :nb] = True
        if nb == 0:
            continue
        w = np.asarray(src_wavs[i])[: nb * frame]                      # ONLY the valid part
        rms = torch.from_numpy(w.reshape(nb, frame).astype(np.float32).__pow__(2).mean(1) ** 0.5)
        peak = float(rms.max())                                        # peak inside the valid region
        thr = max(rel_threshold * peak, abs_floor)
        raw_active[i, :nb] = rms > thr

    valid_source &= valid_mix.unsqueeze(0)
    source_active = raw_active & valid_source
    return dict(valid_mix=valid_mix, valid_source=valid_source, source_active=source_active,
                union_active=source_active.any(0), overlap_active=source_active.all(0))


# ------------------------------------------------------------------ collator --
class LengthCollator:
    """Wrap the existing collator and add `mix_length` / `clean_lengths` (pre-padding)."""

    def __init__(self, base):
        self.base = base

    def __call__(self, items):
        batch = self.base(items)
        # the dataset stores the post-preprocessing waveform; its length IS the valid one
        lengths = [torch.as_tensor(it["clean_lengths"], dtype=torch.long) for it in items
                   if "clean_lengths" in it]
        if lengths:
            batch["clean_lengths"] = torch.stack(lengths)              # (B, S)
            batch["mix_length"] = torch.as_tensor(
                [int(it["mix_length"]) for it in items], dtype=torch.long)
        return batch


def masks_from_batch(batch, n_frames=None):
    """(B,S,T) masks for a whole batch; T defaults to the stream's own time axis."""
    clean = batch["clean_audios"]                     # (B, S, Lp)
    cl = batch["clean_lengths"]                        # (B, S)
    ml = batch["mix_length"]                           # (B,)
    B, S, Lp = clean.shape
    T = n_frames or N_FRAMES
    vm, vs, sa, un, ov = [], [], [], [], []
    for b in range(B):
        m = build_masks([clean[b, i].numpy() for i in range(S)], cl[b], int(ml[b]), n_frames=T)
        vm.append(m["valid_mix"]); vs.append(m["valid_source"]); sa.append(m["source_active"])
        un.append(m["union_active"]); ov.append(m["overlap_active"])
    out = dict(valid_mix=torch.stack(vm), valid_source=torch.stack(vs),
               source_active=torch.stack(sa), union_active=torch.stack(un),
               overlap_active=torch.stack(ov))
    check_masks(out)
    return out


def check_masks(m, T_expected=None):
    """The invariants, with broadcast-correct shapes."""
    sa, vm, vs = m["source_active"], m["valid_mix"], m["valid_source"]
    un, ov = m["union_active"], m["overlap_active"]
    B, S, T = sa.shape
    assert vm.shape == (B, T), (vm.shape, (B, T))
    assert vs.shape == (B, S, T) and un.shape == (B, T) and ov.shape == (B, T)
    if T_expected is not None:
        assert T == T_expected, "feature time axis %d != %d" % (T, T_expected)
    assert not (sa & ~vm[:, None, :]).any(), "activity outside valid_mix"
    assert not (sa & ~vs).any(), "activity outside valid_source"
    assert torch.equal(un, sa.any(dim=1)), "union != any(source_active)"
    assert torch.equal(ov, sa.all(dim=1)), "overlap != all(source_active)"
    assert not (ov & ~un).any(), "overlap not a subset of union"
    return True


# --------------------------------------------------------------------- smoke --
if __name__ == "__main__":
    # synthetic: S=2, the second source is much longer, so the tail is single-speaker
    torch.manual_seed(0)
    Lp = 1500 * FRAME
    s0 = np.zeros(Lp, dtype=np.float32); s0[:1200 * FRAME] = 0.3 * np.random.randn(1200 * FRAME)
    s1 = np.zeros(Lp, dtype=np.float32); s1[:1400 * FRAME] = 0.3 * np.random.randn(1400 * FRAME)
    src = np.stack([s0, s1])
    lens = np.array([1200 * FRAME, 1400 * FRAME])
    mix_len = Lp
    t = build_masks(src, lens, mix_len)
    T = t["valid_mix"].numel()
    m = {k: v.unsqueeze(0) for k, v in t.items()}
    check_masks(m, T_expected=T)
    tail = torch.arange(T) >= 1200
    rec = t["source_active"][1] & tail
    assert bool(rec.any()), "no tail activity"
    assert bool(t["union_active"][rec].all()) and not bool(t["overlap_active"][rec].any())
    print("union %d | overlap %d | union-overlap %d | tail-recovered %d frames %s"
          % (int(t["union_active"].sum()), int(t["overlap_active"].sum()),
             int((t["union_active"] & ~t["overlap_active"]).sum()), int(rec.sum()),
             torch.nonzero(rec).flatten().tolist()[:6]))
    print("ALL MASK INVARIANTS PASSED")
