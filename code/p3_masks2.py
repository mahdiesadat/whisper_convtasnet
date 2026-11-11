"""P3 activity masks using explicit post-preprocessing, pre-padding lengths.

Temporal policy: complete `frame`-sample intervals starting at sample zero.
This is NOT a proof of encoder receptive-field alignment. Inputs must already
be aligned mono 16 kHz waveforms. Outputs stay on the waveform device.
"""
import math
import torch


def _lengths(value, shape, device, name):
    x = torch.as_tensor(value, device=device)
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name}: expected {shape}, got {tuple(x.shape)}")
    if x.dtype == torch.bool or x.is_floating_point() or x.is_complex():
        raise ValueError(f"{name}: integer lengths required")
    if (x < 0).any():
        raise ValueError(f"{name}: negative length")
    return x.to(torch.long).clone()


def build_masks(src_wavs, src_lengths, mix_length, *, n_frames,
                frame=320, rel_threshold=0.1, abs_floor=1e-3):
    """src_wavs: (S,Lp); lengths: (S,) and scalar; outputs on input device.

    n_frames must come from the actual feature time axis. No inferred lengths,
    no default feature length, and no CPU/NumPy conversion are used.
    """
    for name, value in (("n_frames", n_frames), ("frame", frame)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not all(math.isfinite(v) and v >= 0 for v in (rel_threshold, abs_floor)):
        raise ValueError("thresholds must be finite and nonnegative")
    w = torch.as_tensor(src_wavs)
    if w.ndim != 2 or w.shape[0] == 0 or w.is_complex():
        raise ValueError("src_wavs must be real (S,Lp), S > 0")
    S, Lp = w.shape
    lengths = _lengths(src_lengths, (S,), w.device, "src_lengths")
    ml = _lengths(mix_length, (), w.device, "mix_length")
    if (lengths > Lp).any():
        raise ValueError("source length exceeds available waveform")
    grid = torch.arange(n_frames, device=w.device)
    mix_blocks = (ml // frame).clamp(max=n_frames)
    vm = grid < mix_blocks
    blocks = torch.minimum(lengths // frame, mix_blocks)
    vs = grid[None, :] < blocks[:, None]
    active = torch.zeros_like(vs)
    # Labels/selection masks deliberately do not participate in autograd.
    with torch.no_grad():
        for s in range(S):
            nb = int(blocks[s])
            if nb == 0:
                continue
            valid = w[s, :nb * frame].detach().float()
            if not torch.isfinite(valid).all():
                raise ValueError("nonfinite audio inside valid complete intervals")
            rms = valid.reshape(nb, frame).square().mean(1).sqrt()
            if not torch.isfinite(rms).all():
                raise ValueError("nonfinite RMS")
            threshold = (rel_threshold * rms.max()).clamp(min=abs_floor)
            active[s, :nb] = rms > threshold
    sa = active & vs
    return dict(valid_mix=vm, valid_source=vs, source_active=sa,
                union_active=sa.any(0), overlap_active=sa.all(0))


class LengthCollator:
    """Carry dataset-supplied lengths; never infer them from padded waveforms.

    Dataset must provide clean_lengths and mix_length AFTER aligned preprocessing.
    Base collator must preserve sample/source order and temporal coordinates;
    a base that crops, truncates, or prepends audio must update lengths itself.
    """
    def __init__(self, base):
        self.base = base

    def __call__(self, items):
        if not items:
            raise ValueError("empty batch")
        cls, mls = [], []
        for i, item in enumerate(items):
            if "clean_lengths" not in item or "mix_length" not in item:
                raise ValueError(f"item {i}: missing explicit valid lengths")
            raw = torch.as_tensor(item["clean_lengths"])
            if raw.ndim != 1 or raw.numel() == 0:
                raise ValueError(f"item {i}: clean_lengths must be nonempty (S,)")
            cls.append(_lengths(raw, raw.shape, "cpu", "clean_lengths"))
            mls.append(_lengths(item["mix_length"], (), "cpu", "mix_length"))
        if any(x.shape != cls[0].shape for x in cls):
            raise ValueError("inconsistent source counts")
        # Capture and clone metadata BEFORE base can mutate its inputs.
        cl, ml = torch.stack(cls), torch.stack(mls)
        batch = self.base(items)
        for key, value in (("clean_lengths", cl), ("mix_length", ml)):
            if key in batch and not torch.equal(torch.as_tensor(batch[key]).cpu(), value):
                raise ValueError(f"base collator changed {key}; audit preprocessing")
            batch[key] = value
        clean = batch["clean_audios"]
        if clean.ndim != 3 or tuple(clean.shape[:2]) != tuple(cl.shape):
            raise ValueError("clean_audios must preserve (B,S,Lp)")
        if (cl > clean.shape[-1]).any():
            raise ValueError("base truncated sources without updating valid lengths")
        return batch


def masks_from_batch(batch, *, n_frames, **kwargs):
    clean = torch.as_tensor(batch["clean_audios"])
    if clean.ndim != 3 or min(clean.shape[:2]) == 0:
        raise ValueError("clean_audios must be (B,S,Lp), B,S > 0")
    B, S, _ = clean.shape
    cl = _lengths(batch["clean_lengths"], (B, S), clean.device, "clean_lengths")
    ml = _lengths(batch["mix_length"], (B,), clean.device, "mix_length")
    rows = [build_masks(clean[b], cl[b], ml[b], n_frames=n_frames, **kwargs)
            for b in range(B)]
    out = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
    check_masks(out, T_expected=n_frames)
    return out


def check_masks(m, *, T_expected):
    sa, vm, vs = m["source_active"], m["valid_mix"], m["valid_source"]
    un, ov = m["union_active"], m["overlap_active"]
    assert sa.ndim == 3
    B, S, T = sa.shape
    assert B > 0 and S > 0 and T == T_expected
    assert all(x.dtype == torch.bool for x in m.values())
    assert vm.shape == (B, T) and vs.shape == (B, S, T)
    assert un.shape == ov.shape == (B, T)
    assert not (vs & ~vm[:, None, :]).any()
    assert not (sa & ~vm[:, None, :]).any()
    assert not (sa & ~vs).any()
    assert torch.equal(un, sa.any(1))
    assert torch.equal(ov, sa.all(1))
    assert not (ov & ~un).any()
    return True
