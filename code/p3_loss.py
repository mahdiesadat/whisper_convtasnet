"""P3 loss, revision 4 â€” FINAL helper (the patcher copies this file as `p3_loss.py`).

Delta vs revision 3 (review of 2026-11-11):
  1. ONE shared helper `_effective_mask()` builds the boolean mask, and BOTH the loss
     statistics and `measure_delta` report on that effective mask (rev 3 still reported
     Delta/hinge on the pre-guard `active` mask);
  2. sample counters split before/after the guard:
       samples_without_overlap | samples_with_effective_overlap
       samples_without_effective_overlap | samples_all_frames_filtered
     N_b and K in the documented reduction are defined AFTER the guard, so a sample that
     lost every frame does not take part in the final mean;
  3. `measure_delta` uses the effective mask for stream_pair_cos and cancellation;
  4. test16 compares the loss against a run with that frame removed from BOTH branches
     and checks that both branches of the frame get zero gradient;
  5. wording: the auxiliary gradient reaches the parameters through the chain rule and
     the factor beta; the final update depends on the summed gradients, clipping, the
     optimizer and its state, and the learning rate (no closed-form claim).

Guards, documented honestly
---------------------------
* HARD, PAIRWISE exclusion of comparisons whose centred vectors are tiny (not a soft
  regulariser); a frame is dropped for BOTH branches when either fails, so den == 2*N_b;
* it only protects against tiny-but-FINITE norms: NaN * 0 is still NaN, so it is not a
  general NaN/Inf guard;
* MIN_NORM = 1e-6 is an experimental threshold, not a tuned value.

Math (post-guard; M_b = effective overlap mask of sample b)
    u_i,t = h_t + alpha * s_i,t                      (the stream Whisper consumes)
    u~ = u - c , z~tgt = z_tgt - c , z~oth = z_oth - c          (c fixed, no grad)
    d  = cos(u~, z~tgt) - cos(u~, z~oth)                        d in [-2, 2]
    L  = (1/K) sum_{b: N_b>0} 1/(2 N_b) sum_i sum_{t in M_b} [m - d]^+
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

EPS = 1e-8
MIN_NORM = 1e-6


@dataclass
class CenterStats:
    vector: torch.Tensor          # (C,) float32 CPU
    n_frames: int                 # counted over B*S rows (both branches)
    n_utterances: int
    depth: str
    checkpoint: str
    train_manifest: str
    helper_sha256: str = ""
    skip_padding: bool = True
    silence_policy: str = "frames whose mixture rms is 0 are excluded"


def accumulate_center_sum(stream, mix_active=None) -> torch.Tensor:
    """stream (B*S, C, T) -> sum over frames (C,).  Never touches the graph."""
    if stream.dim() != 3:
        raise ValueError("stream must be (B*S, C, T), got %s" % (tuple(stream.shape),))
    x = stream.detach().float().permute(0, 2, 1)
    if mix_active is not None:
        if mix_active.shape != x.shape[:2]:
            raise ValueError("mix_active must be (B*S, T) = %s, got %s"
                             % (tuple(x.shape[:2]), tuple(mix_active.shape)))
        w = mix_active.to(x.device).bool().float().unsqueeze(-1)
        return (x * w).sum(dim=(0, 1))
    return x.sum(dim=(0, 1))


def finalize_center(total_sum, n_frames) -> torch.Tensor:
    if n_frames <= 0:
        raise ValueError("cannot build a centre from zero frames")
    return (total_sum / float(n_frames)).to(torch.float32).cpu()


def frame_activity(wav, n_frames: int = 1500, frame_len: int = 320,
                   rel_threshold: float = 0.1, abs_floor: float = 1e-3):
    """Approximate speech activity on the FEATURE time grid.

    Whisper-small: mel hop 160 at 16 kHz -> 10 ms per mel frame; two convolutions with
    temporal downsampling by stride 2 in the SECOND -> 20 ms per feature frame = 320
    samples.  Alignment in the sense that feature frame t advances by the same 20 ms;
    NOT an exact temporal-support match (windowed STFT, conv padding, attention field).
    """
    target = n_frames * frame_len
    if wav.shape[-1] < target:
        wav = F.pad(wav, (0, target - wav.shape[-1]))
    elif wav.shape[-1] > target:
        wav = wav[..., :target]
    frames = wav.reshape(*wav.shape[:-1], n_frames, frame_len)
    rms = frames.float().pow(2).mean(dim=-1).sqrt()
    peak = rms.amax(dim=-1, keepdim=True).clamp(min=abs_floor)
    return rms > torch.clamp(rel_threshold * peak, min=abs_floor)


def assignment_from_pit(perms, loss_min_idx, num_spks: int):
    """Reference index for every stream, from the CE's chosen permutation."""
    if num_spks != 2:
        raise ValueError("P3 supports 2 speakers only (got %d)" % num_spks)
    perms = [tuple(int(x) for x in p) for p in perms]
    if sorted(perms) != sorted(itertools.permutations(range(num_spks))):
        raise ValueError("perms must enumerate all permutations of range(num_spks)")
    idx = loss_min_idx.detach().long()
    table = torch.tensor(perms, dtype=torch.long, device=idx.device)
    return torch.argsort(table, dim=1)[idx]


def _check_inputs(stream, clean, assignment, active, center):
    if stream.dim() != 4:
        raise ValueError("stream must be (B,S,T,C), got %s" % (tuple(stream.shape),))
    b, s, t, d = stream.shape
    if s != 2:
        raise ValueError("P3 supports 2 speakers only (stream has S=%d)" % s)
    if clean.shape != stream.shape:
        raise ValueError("clean %s must match stream %s" % (tuple(clean.shape), tuple(stream.shape)))
    if clean.device != stream.device:
        raise ValueError("clean and stream must be on the same device")
    if active.device != stream.device:
        raise ValueError("active and stream must be on the same device")
    if assignment.dtype != torch.long:
        raise ValueError("assignment must be torch.long, got %s" % assignment.dtype)
    if assignment.device != stream.device:
        raise ValueError("assignment on %s but stream on %s" % (assignment.device, stream.device))
    if assignment.shape != (b, s):
        raise ValueError("assignment must be (B,S)=%s, got %s" % ((b, s), tuple(assignment.shape)))
    if not torch.equal(assignment.sort(dim=1).values,
                       torch.arange(s, device=assignment.device).expand(b, s)):
        raise ValueError("every assignment row must be a permutation of [0,1]")
    if active.dtype != torch.bool:
        raise ValueError("active must be bool, got %s" % active.dtype)
    if active.shape != (b, s, t):
        raise ValueError("active must be (B,S,T)=%s, got %s" % ((b, s, t), tuple(active.shape)))
    if not torch.equal(active[:, 0], active[:, 1]):
        raise ValueError("P3 expects a shared overlap mask for both branches "
                         "(the reduction assumes den = 2*N_b)")
    if center is not None:
        if not torch.is_tensor(center):
            raise ValueError("center must be a tensor or None")
        if center.shape != (d,):
            raise ValueError("center must have shape (%d,), got %s" % (d, tuple(center.shape)))


def _centered_triplet(stream, clean, assignment, center):
    b, s, t, d = stream.shape
    idx = assignment[:, :, None, None].expand(b, s, t, d)
    tgt = clean.detach().gather(1, idx).float()
    oth = tgt.flip(dims=[1])
    u = stream.float()
    if center is not None:
        c = center.detach().to(device=stream.device, dtype=torch.float32).view(1, 1, 1, d)
        u, tgt, oth = u - c, tgt - c, oth - c
    return u, tgt, oth


def _effective_mask(u, tgt, oth, active):
    """The ONE place the guard is defined, shared by the loss and the diagnostics."""
    with torch.no_grad():
        branch_ok = ((u.norm(dim=-1) > MIN_NORM) & (tgt.norm(dim=-1) > MIN_NORM)
                     & (oth.norm(dim=-1) > MIN_NORM))
        overlap_before = active[:, 0]
        effective = overlap_before & branch_ok.all(dim=1)
    return effective[:, None, :].expand_as(active), overlap_before


@torch.no_grad()
def measure_delta(stream, clean, assignment, active, center):
    """Delta statistics on the EFFECTIVE mask (the same guard as the loss)."""
    _check_inputs(stream, clean, assignment, active, center)
    u, tgt, oth = _centered_triplet(stream, clean, assignment, center)
    pos = F.cosine_similarity(u, tgt, dim=-1, eps=EPS)
    neg = F.cosine_similarity(u, oth, dim=-1, eps=EPS)
    d = pos - neg
    eff_mask, overlap_before = _effective_mask(u, tgt, oth, active)
    pair = eff_mask[:, 0]
    before_has = overlap_before.any(dim=1)
    after_has = pair.any(dim=1)
    counters = dict(
        overlap_frames_before_guard=int(overlap_before.sum()),
        overlap_frames_after_guard=int(pair.sum()),
        frames_skipped_by_norm_guard=int((overlap_before & ~pair).sum()),
        samples_without_overlap=int((~before_has).sum()),
        samples_with_effective_overlap=int(after_has.sum()),
        samples_without_effective_overlap=int((~after_has).sum()),
        samples_all_frames_filtered=int((before_has & ~after_has).sum()),
    )
    if int(pair.sum()) == 0:
        out = dict(n=0, stream_pair_cos=None, delta_pair_sum_abs_mean=None)
        out.update(counters)          # counters are reported even with no frames
        return out
    dv = d[eff_mask]
    pair_cos = F.cosine_similarity(u[:, 0], u[:, 1], dim=-1, eps=EPS)
    return dict(
        n=int(dv.numel()),
        mean=float(dv.mean()), mean_abs=float(dv.abs().mean()), std=float(dv.std()),
        min=float(dv.min()), max=float(dv.max()),
        frac_positive=float((dv > 0).float().mean()),
        pos_mean=float(pos[eff_mask].mean()), neg_mean=float(neg[eff_mask].mean()),
        quantiles={p: float(torch.quantile(dv, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
        per_branch=dict(delta=[float(d[:, i][eff_mask[:, i]].mean()) for i in (0, 1)],
                        pos=[float(pos[:, i][eff_mask[:, i]].mean()) for i in (0, 1)],
                        neg=[float(neg[:, i][eff_mask[:, i]].mean()) for i in (0, 1)]),
        stream_pair_cos=float(pair_cos[pair].mean()),
        delta_pair_sum_abs_mean=float((d[:, 0] + d[:, 1])[pair].abs().mean()),
        **counters,
    )


def p3_margin_loss(stream, clean, assignment, active, center, margin):
    """(loss, stats).  Samples without effective overlap contribute exactly 0."""
    _check_inputs(stream, clean, assignment, active, center)
    if not math.isfinite(margin):
        raise ValueError("margin must be finite")

    b, s, t, d = stream.shape
    u, tgt, oth = _centered_triplet(stream, clean, assignment, center)
    delta = (F.cosine_similarity(u, tgt, dim=-1, eps=EPS)
             - F.cosine_similarity(u, oth, dim=-1, eps=EPS))
    hinge = F.relu(margin - delta)

    eff_mask, overlap_before = _effective_mask(u, tgt, oth, active)
    m = eff_mask.to(hinge.dtype)
    num = (hinge * m).sum(dim=(1, 2))
    den = m.sum(dim=(1, 2))                  # == 2 * N_b, N_b counted AFTER the guard
    has = den > 0
    per_sample = num / den.clamp(min=1.0)
    loss = per_sample[has].mean() if bool(has.any()) else (u.sum() * 0.0)

    with torch.no_grad():
        dv = delta.detach()[eff_mask]
        hv = hinge.detach()[eff_mask]
        before_has = overlap_before.any(dim=1)
        after_has = eff_mask[:, 0].any(dim=1)
        stats = dict(
            effective_overlap_frames=int(eff_mask[:, 0].sum().item()),
            overlap_frames_before_guard=int(overlap_before.sum().item()),
            overlap_frames_after_guard=int(eff_mask[:, 0].sum().item()),
            frames_skipped_by_norm_guard=int((overlap_before & ~eff_mask[:, 0]).sum().item()),
            samples_without_overlap=int((~before_has).sum().item()),
            samples_with_effective_overlap=int(after_has.sum().item()),
            samples_without_effective_overlap=int((~after_has).sum().item()),
            samples_all_frames_filtered=int((before_has & ~after_has).sum().item()),
            hinge_active=float((hv > 0).float().mean().item()) if hv.numel() else 0.0,
            delta_mean=float(dv.mean().item()) if dv.numel() else float("nan"),
            delta_mean_abs=float(dv.abs().mean().item()) if dv.numel() else float("nan"),
            delta_frac_positive=float((dv > 0).float().mean().item()) if dv.numel() else float("nan"),
            delta_pair_sum_abs_mean=float((delta.detach()[:, 0] + delta.detach()[:, 1])[eff_mask[:, 0]].abs().mean().item())
            if bool(eff_mask[:, 0].any()) else float("nan"),
        )
    return loss, stats


def _run_tests():
    B, S, T, C = 2, 2, 5, 8
    perms = list(itertools.permutations(range(S)))

    def mk(seed):
        return torch.randn(B, S, T, C, generator=torch.Generator().manual_seed(seed))

    stream, clean = mk(1), mk(2)
    center = torch.zeros(C)
    active = torch.ones(B, S, T, dtype=torch.bool)
    a_id = assignment_from_pit(perms, torch.tensor([0, 0]), S)
    l1, _ = p3_margin_loss(stream, clean, a_id, active, center, 0.2)
    print("test1  identity assignment                     ok  loss=%.5f" % l1)

    a_sw = assignment_from_pit(perms, torch.tensor([1, 1]), S)
    l2, _ = p3_margin_loss(stream, clean, a_sw, active, center, 0.2)
    assert not torch.allclose(l1, l2)
    print("test2  assignment sensitivity (not invariance) ok  loss=%.5f" % l2)

    l3, _ = p3_margin_loss(stream, clean.flip(dims=[1]), a_sw, active, center, 0.2)
    assert torch.allclose(l3, l1, atol=1e-6)
    print("test3  invariance: refs swapped + asg          ok  %.5f == %.5f" % (l3, l1))

    l4, _ = p3_margin_loss(stream.flip(dims=[1]), clean, a_sw, active, center, 0.2)
    assert torch.allclose(l4, l1, atol=1e-6)
    print("test4  invariance: streams swapped + asg       ok  %.5f == %.5f" % (l4, l1))

    act0 = torch.zeros(B, S, T, dtype=torch.bool)
    s5 = stream.clone().requires_grad_(True)
    l5, st5 = p3_margin_loss(s5, clean, a_id, act0, center, 0.2)
    l5.backward()
    assert float(l5) == 0.0 and float(s5.grad.abs().sum()) == 0.0
    print("test5  no overlap: 0.0 and zero grad           ok  (before=%d)"
          % st5["samples_without_overlap"])

    pad = 3
    ps = torch.cat([stream, torch.randn(B, S, pad, C, generator=torch.Generator().manual_seed(9))],
                   dim=2).requires_grad_(True)
    pc = torch.cat([clean, torch.randn(B, S, pad, C, generator=torch.Generator().manual_seed(10))], dim=2)
    pa = torch.cat([active, torch.zeros(B, S, pad, dtype=torch.bool)], dim=2)
    l6, _ = p3_margin_loss(ps, pc, a_id, pa, center, 0.2)
    l6.backward()
    assert torch.allclose(l6, l1, atol=1e-6)
    assert float(ps.grad[:, :, -pad:, :].abs().sum()) == 0.0
    print("test6  temporal padding invariance             ok  %.5f == %.5f, pad grad=0" % (l6, l1))

    c2 = torch.randn(C, requires_grad=True)
    u, zt, zo = _centered_triplet(stream, clean, a_id, c2)
    cc = c2.detach().view(1, 1, 1, -1)
    assert torch.allclose(u, stream.float() - cc)
    assert torch.allclose(zt, clean.float().gather(1, a_id[:, :, None, None].expand(B, S, T, C)) - cc)
    assert torch.allclose(zo, zt.flip(1))
    print("test7  centre on u, z_tgt AND z_oth            ok")

    s8 = stream.clone().requires_grad_(True)
    p3_margin_loss(s8, clean, a_id, active, c2, 0.2)[0].backward()
    assert c2.grad is None or float(c2.grad.abs().sum()) == 0.0
    print("test8  centre gets no gradient                 ok")

    s9, c9 = stream.clone().requires_grad_(True), clean.clone().requires_grad_(True)
    p3_margin_loss(s9, c9, a_id, active, center, 0.2)[0].backward()
    assert float(s9.grad.abs().sum()) > 0
    assert c9.grad is None or float(c9.grad.abs().sum()) == 0.0
    print("test9  gradient: stream yes, clean no          ok")

    _, s_lo = p3_margin_loss(stream, clean, a_id, active, center, -5.0)
    _, s_hi = p3_margin_loss(stream, clean, a_id, active, center, +5.0)
    assert s_lo["hinge_active"] == 0.0 and s_hi["hinge_active"] == 1.0
    print("test10 margin controls the hinge               ok")

    max_g = 0.0
    for scale in (0.0, 1e-12, 1e-6):
        tiny = (torch.randn(B, S, T, C, generator=torch.Generator().manual_seed(21)) * scale).requires_grad_(True)
        ref = torch.randn(B, S, T, C, generator=torch.Generator().manual_seed(22)) * max(scale, 1e-6)
        lt, _ = p3_margin_loss(tiny, ref, a_id, active, center, 0.2)
        lt.backward()
        assert torch.isfinite(lt) and torch.isfinite(tiny.grad).all()
        max_g = max(max_g, float(tiny.grad.abs().max()))
    print("test11 backward at scale 0/1e-12/1e-6          ok  max|grad|=%.3e" % max_g)

    try:
        p3_margin_loss(torch.randn(B, 3, T, C), torch.randn(B, 3, T, C),
                       torch.zeros(B, 3, dtype=torch.long),
                       torch.ones(B, 3, T, dtype=torch.bool), torch.zeros(C), 0.2)
        raise AssertionError("S=3 must be rejected")
    except ValueError:
        print("test12 S!=2 rejected inside the loss           ok")

    wz = torch.zeros(16000 * 2)
    tone = torch.sin(torch.linspace(0, 400 * math.pi, 16000 * 2)) * 0.5
    assert int(frame_activity(wz).sum()) == 0
    at = frame_activity(tone)
    assert at.shape[-1] == 1500 and int(at.sum()) > 0
    print("test13 activity mask                           ok  tone %d/1500" % int(at.sum()))

    d = measure_delta(stream, clean, a_id, active, center)
    assert d["n"] == B * S * T and len(d["per_branch"]["delta"]) == S
    print("test14 SYNTHETIC stats (random tensors, NOT Whisper features; a LARGE mean|d0+d1| means the branches do NOT cancel)")
    print("       mean=%+.4f mean|d|=%.4f pair_cos=%.3f  mean|d0+d1|=%.4f"
          % (d["mean"], d["mean_abs"], d["stream_pair_cos"], d["delta_pair_sum_abs_mean"]))

    act_bad = active.clone()
    act_bad[0, 1, 0] = False
    try:
        p3_margin_loss(stream, clean, a_id, act_bad, center, 0.2)
        raise AssertionError("a per-branch mask must be rejected")
    except ValueError:
        print("test15 shared overlap mask enforced            ok")

    s16 = stream.clone()
    s16[0, 1, 0] = 0.0
    l16, st16 = p3_margin_loss(s16, clean, a_id, active, center, 0.2)
    expected_active = active.clone()
    expected_active[0, :, 0] = False
    l16_ref, _ = p3_margin_loss(stream, clean, a_id, expected_active, center, 0.2)
    assert torch.allclose(l16, l16_ref, atol=1e-6), (l16, l16_ref)
    assert st16["frames_skipped_by_norm_guard"] == 1
    s16g = s16.clone().requires_grad_(True)
    p3_margin_loss(s16g, clean, a_id, active, center, 0.2)[0].backward()
    assert float(s16g.grad[0, :, 0, :].abs().sum()) == 0.0
    print("test16 one branch collapses -> frame removed for BOTH  ok  (loss matches, grad=0)")

    s17 = torch.zeros(B, S, T, C, requires_grad=True)
    l17, st17 = p3_margin_loss(s17, torch.zeros(B, S, T, C), a_id, active, center, 0.2)
    l17.backward()
    assert float(l17) == 0.0 and float(s17.grad.abs().sum()) == 0.0
    assert st17["samples_all_frames_filtered"] == B
    assert st17["samples_without_overlap"] == 0
    print("test17 all frames filtered -> 0.0, grad 0, counted as filtered  ok")

    print("\nALL P3 UNIT TESTS PASSED (17)")


if __name__ == "__main__":
    _run_tests()


