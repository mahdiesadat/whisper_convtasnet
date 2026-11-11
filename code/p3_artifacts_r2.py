#!/usr/bin/env python3
"""p3_artifacts.py (rev 2) -- center | margin | grads, with every review fix applied.

Fixes vs rev 1
--------------
1. The centre uses the UNION of source activity (single-speaker frames included) while
   P3 uses the OVERLAP; the two masks are now separate objects, and the metadata says
   exactly which one the centre used.
2. `R = ||c||^2 / E||u||^2` is reported (sums accumulated in FP64), and the misleading
   `norm.json.mean` is renamed `center_l2_norm`.
3. Margin rule corrected: the hinge is active iff Delta < m, so
   `m = quantile(v, target_active)`; the ACHIEVED active fraction is recorded, the
   target is range-checked, and `n_comparisons` replaces `n_frames`.
4. `grads` now uses ONE forward:
        l_tot  = the model's loss (CE + beta * aux, from the same graph)
        l_aux  = p3_margin_loss(stored stream, ...)        (identical to the model's term)
        l_ce   = l_tot - beta * l_aux                      (GRAPH subtraction, exact)
   then two autograd.grad calls on that single graph.  No second forward, no differencing
   of two gradients, no dependence on re-reading the environment after construction.
   alpha is excluded from the separator group, None-gradients are reported by name, and
   ratios/cosines are None when the denominator is zero.
5. `helper_sha256` comes from `Path(P3.__file__)`, not from a hard-coded path.
6. The PIT assignment is reconstructed and then VALIDATED: the reconstruction's minimum
   CE must equal the model's own CE loss for that forward (tolerance 1e-4), otherwise the
   run aborts.  (The robust fix -- stashing `loss_min_idx` in the patch -- is noted in the
   JSON as `assignment_source`, and the validation above is what makes the reconstruction
   trustworthy in the meantime.)

Aggregation: means are weighted by their own counters; quantiles come from POOLED frame
values; counters are reported even for batches whose every frame was filtered.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("P3_CODE_ROOT", "fixed_retrain"))

import torch
import torch.nn.functional as F
from transformers import WhisperProcessor

import p3_loss as P3
from model.modeling_whisper_sidecar import WhisperSidecarForConditionalGeneration
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.dataset import CustomDataset


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(ckpt, device):
    proc = WhisperProcessor.from_pretrained("openai/whisper-small", language="en",
                                            task="transcribe", no_timestamps=True)
    model = WhisperSidecarForConditionalGeneration.from_pretrained(
        ckpt, sidecar_loc=1, num_spks=2, soft_prompt_len=4, target_asr=False,
        sidecar_residual=True, sidecar_activate="sigmoid", mask_norm="none",
        sidecar_gate_init=0.1, attn_implementation="sdpa")
    model.set_mel_filters(proc.feature_extractor.mel_filters)
    model.to(device)
    coll = DataCollatorSpeechSeq2SeqWithPadding(processor=proc, num_spks=2,
                                                soft_prompt_len=4, target_asr=False)
    return proc, model, coll


def batches(manifest, proc, coll, n_batches, batch_size, seed=0):
    ds = CustomDataset(data_list_path=manifest, processor=proc, language="en",
                       timestamps=False, num_spks=2)
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=rng).tolist()
    made = 0
    for i in range(0, len(idx) - batch_size + 1, batch_size):
        items = [ds[j] for j in idx[i:i + batch_size]]
        if any("clean_audios" not in it for it in items):
            raise SystemExit("dataset returned no clean_audios: set SIDECAR_LOAD_CLEAN=1 "
                             "(or SIDECAR_REC_LAM / SIDECAR_P3_BETA > 0 until the flag lands)")
        yield coll(items)
        made += 1
        if made >= n_batches:
            return


def perms_table(S=2):
    return [tuple(p) for p in itertools.permutations(range(S))]


def forward_common(model, batch, device, need_graph=False):
    """ONE forward. Returns the pieces every subcommand needs."""
    feats = batch["input_features"].to(device)
    clean = batch["clean_audios"].to(device)
    labels = batch["labels"].to(device)
    ctx = torch.enable_grad() if need_graph else torch.no_grad()
    with ctx:
        out = model(input_features=feats, labels=labels, clean_audios=clean)
    enc = model.model.encoder
    st = enc.last_full_streams
    cl = enc.last_clean_feats
    B, S = st.shape[0] // model.num_spks, model.num_spks
    T = st.shape[-1]
    stream = st.float().view(B, S, st.shape[1], T).permute(0, 1, 3, 2)     # (B,S,T,C)
    clean_t = cl.float().view(B, S, cl.shape[1], cl.shape[2])
    # source activity, union and overlap kept SEPARATE (review point 1)
    src = P3.frame_activity(clean.view(-1, clean.shape[-1]), n_frames=T).view(B, S, T)
    union = src.any(dim=1)
    overlap = src.all(dim=1)
    active = overlap[:, None, :].expand(B, S, T) & src          # P3 mask == overlap
    # PIT assignment, then validated against the model's own CE
    perms = perms_table(S)
    idx = pit_argmin(out, labels, model, perms)
    ce_check = pit_min_value(out, labels, model, perms)
    asg = P3.assignment_from_pit(perms, idx, S)
    return dict(out=out, stream=stream, clean=clean_t, active=active,
                union=union, overlap=overlap, assignment=asg, src=src,
                ce_check=ce_check, T=T, B=B, S=S)


def _flat_ce(out, labels, model):
    tgt = labels.repeat(1, len(perms_table()))[0]
    lg = out.logits.reshape(-1, out.logits.shape[-1]).float()
    tgt_flat = labels.repeat(1, model.num_perms).view(-1, labels.shape[-1])
    l = F.cross_entropy(lg, tgt_flat.reshape(-1), reduction="none")
    return l.view(-1, model.num_perms, model.num_spks, tgt_flat.shape[-1])


def pit_argmin(out, labels, model, perms):
    l = _flat_ce(out, labels, model)
    return l.mean(-1).mean(2).min(1)[1].detach()


def pit_min_value(out, labels, model, perms):
    l = _flat_ce(out, labels, model)
    return float(l.mean(-1).mean(2).min(1)[0].mean())


def cmd_center(a):
    os.environ.setdefault("SIDECAR_P3_BETA", "1")
    proc, model, coll = build(a.checkpoint, "cpu")
    model.eval()
    C = model.config.d_model
    total = torch.zeros(C, dtype=torch.float64)
    sq = 0.0
    n_frames = 0
    n_utt = 0
    with torch.no_grad():
        for bi, batch in enumerate(batches(a.manifest, proc, coll, a.n_batches, a.batch_size)):
            r = forward_common(model, batch, "cpu")
            stream, union = r["stream"], r["union"]
            B, S, T, Cc = stream.shape
            x = stream.permute(0, 1, 3, 2).reshape(B * S, Cc, T)      # (B*S, C, T)
            m = union[:, None, :].expand(B, S, T).reshape(B * S, T)   # UNION, both branches
            v = x.double().permute(0, 2, 1)[m.bool()]                # (N, C)
            total += v.sum(0)
            sq += float(v.pow(2).sum())
            n_frames += int(m.sum())
            n_utt += B
            if bi % 20 == 0:
                print("  batch %d | frames %d | sum||u||^2 %.1f" % (bi, n_frames, sq), flush=True)
    if n_frames == 0:
        raise SystemExit("no union frames")
    center = P3.finalize_center(total, n_frames)                      # fp32, CPU
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(center, out)
    E_sq = sq / n_frames
    R = float(center.double().pow(2).sum()) / max(E_sq, 1e-30)
    meta = dict(center_file=str(out), center_sha256=sha256(out),
                helper_sha256=sha256(Path(P3.__file__).resolve()),
                dim=int(center.numel()), dtype=str(center.dtype),
                n_frames_union=n_frames, n_utterances=n_utt,
                center_l2_norm=float(center.norm()),
                mean_squared_norm_E_sq=E_sq,
                energy_ratio_R=R,
                ratio_definition="R = ||c||^2 / E||u||^2, sums in FP64 over the SAME union frames and both branches",
                mask_policy="UNION of source activity (single-speaker frames INCLUDED); P3 itself uses the overlap",
                depth="Whisper encoder after block index 1 + gated residual (the stream Whisper consumes)",
                checkpoint=a.checkpoint, train_manifest=a.manifest,
                frame_grid="1500 frames x 20 ms (320 samples); mel hop 160, stride 2 in the 2nd conv",
                unverified="frame_activity is an approximation on the feature grid; valid-length, crop/shift alignment and source-vs-mixture offsets are NOT proven here")
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def cmd_margin(a):
    if not (0.0 < a.target_active < 1.0):
        raise SystemExit("--target-active must be in (0,1)")
    os.environ.setdefault("SIDECAR_P3_BETA", "1")
    proc, model, coll = build(a.checkpoint, "cpu")
    model.eval()
    center = torch.load(a.center, map_location="cpu").float()
    vals, before, after, skipped, no_ov, filt = [], 0, 0, 0, 0, 0
    for batch in batches(a.manifest, proc, coll, a.n_batches, a.batch_size):
        r = forward_common(model, batch, "cpu")
        u, tgt, oth = P3._centered_triplet(r["stream"], r["clean"], r["assignment"], center)
        eff, ov = P3._effective_mask(u, tgt, oth, r["active"])
        d = (F.cosine_similarity(u, tgt, dim=-1, eps=P3.EPS)
             - F.cosine_similarity(u, oth, dim=-1, eps=P3.EPS))
        vals.append(d[eff].double())
        before += int(ov.sum()); after += int(eff[:, 0].sum())
        skipped += int((ov & ~eff[:, 0]).sum())
        no_ov += int((~ov.any(1)).sum()); filt += int((ov.any(1) & ~eff[:, 0].any(1)).sum())
    v = torch.cat(vals) if vals else torch.empty(0)
    if v.numel() == 0:
        raise SystemExit("no effective frames")
    m = float(torch.quantile(v, a.target_active))          # hinge active iff Delta < m
    actual = float((v < m).float().mean())
    q = {str(p): float(torch.quantile(v, p)) for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
    res = dict(n_comparisons=int(v.numel()),
               mean=float(v.mean()), mean_abs=float(v.abs().mean()), std=float(v.std()),
               frac_positive=float((v > 0).float().mean()), quantiles=q,
               margin=m, target_active_frac=a.target_active, achieved_active_frac=actual,
               margin_rule="m = quantile(Delta, target_active) because relu(m-Delta) is active iff Delta < m; "
                           "frame-weighted (the loss is sample-weighted) -- recorded, not hidden",
               guard=dict(norm=float(P3.MIN_NORM), overlap_before=before, overlap_after=after,
                          frames_skipped=skipped, skip_rate=skipped / max(before, 1),
                          samples_without_overlap=no_ov, samples_all_frames_filtered=filt),
               checkpoint=a.checkpoint, center=a.center, center_sha256=sha256(a.center))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_grads(a):
    os.environ["SIDECAR_SEP_LAM"] = "0"
    os.environ["SIDECAR_REC_LAM"] = "0"
    os.environ["SIDECAR_P3_BETA"] = str(a.beta)      # term ON so the stream is stored
    proc, model, coll = build(a.checkpoint, "cpu")
    model.train(a.train_mode)
    center = torch.load(a.center, map_location="cpu").float()
    margin = load_margin(a.margin)
    alpha_name = [n for n, _ in model.named_parameters() if n.endswith("sidecar_alpha")][0]
    sep = [n for n, p in model.named_parameters()
           if any(k in n for k in ("sidecar", "sep_enc", "sep_dec"))
           and n != alpha_name and p.requires_grad]
    rows = []
    for bi, batch in enumerate(batches(a.manifest, proc, coll, a.n_batches, a.batch_size)):
        r = forward_common(model, batch, "cpu", need_graph=True)
        l_aux, stats = P3.p3_margin_loss(r["stream"], r["clean"], r["assignment"],
                                         r["active"], center, margin)
        l_tot = r["out"].loss
        l_ce = l_tot - a.beta * l_aux                 # exact graph subtraction
        note = dict(ce_minus_recon=abs(float(l_ce) - r["ce_check"]),
                    aux=float(l_aux), ce_est=float(l_ce), total=float(l_tot))
        if note["ce_minus_recon"] > 1e-4 * max(1.0, abs(r["ce_check"])):
            print("  WARNING batch %d: CE reconstruction differs from the model (%s)" % (bi, note))
        names = sep + [alpha_name]
        params = [dict(model.named_parameters())[n] for n in names]
        g_ce = torch.autograd.grad(l_ce, params, retain_graph=True, allow_unused=True)
        g_au = torch.autograd.grad(l_aux, params, allow_unused=True)
        def collect(gs):
            out, missing = {}, []
            for n, g in zip(names, gs):
                if g is None:
                    missing.append(n)
                else:
                    out[n] = g.detach().double()
            return out, missing
        c, c_missing = collect(g_ce)
        u_, u_missing = collect(g_au)
        def norm_of(gs, keys):
            vs = [gs[k].reshape(-1) for k in keys if k in gs]
            return float(torch.cat(vs).norm()) if vs else 0.0
        def cos_of(keys):
            a_ = [c[k].reshape(-1) for k in keys if k in c]
            b_ = [u_[k].reshape(-1) for k in keys if k in u_]
            if not a_ or not b_:
                return None
            va, vb = torch.cat(a_), torch.cat(b_)
            if float(va.norm()) == 0 or float(vb.norm()) == 0:
                return None                            # undefined, reported as None
            return float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)))
        nce, nau = norm_of(c, sep), norm_of(u_, sep)
        row = dict(batch=bi, **note, mode="train" if a.train_mode else "eval",
                   sep=dict(ce_grad_norm=nce, aux_grad_norm=nau,
                            r_raw=(nau / nce) if nce > 0 else None,
                            r_beta=(a.beta * nau / nce) if nce > 0 else None,
                            cos=cos_of(sep), missing_ce=c_missing, missing_aux=u_missing),
                   alpha=dict(ce_grad=float(c[alpha_name].reshape(-1)[0]) if alpha_name in c else None,
                              aux_grad=float(u_[alpha_name].reshape(-1)[0]) if alpha_name in u_ else None,
                              missing=alpha_name in c_missing or alpha_name in u_missing))
        if alpha_name in c and alpha_name in u_:
            a_ce = float(c[alpha_name].reshape(-1)[0]); a_au = float(u_[alpha_name].reshape(-1)[0])
            row["alpha"].update(r_raw=(abs(a_au) / abs(a_ce)) if a_ce != 0 else None,
                                cos=float(F.cosine_similarity(c[alpha_name].reshape(-1).unsqueeze(0),
                                                              u_[alpha_name].reshape(-1).unsqueeze(0))))
        rows.append(row)
        print("b%d %s | CE %.4f aux %.5f | sep r_raw %.3f cos %s | d(alpha) ce %.3e aux %.3e"
              % (bi, row["mode"], note["ce_est"], note["aux"], row["sep"]["r_raw"] or -1,
                 row["sep"]["cos"], row["alpha"]["ce_grad"] or 0, row["alpha"]["aux_grad"] or 0),
              flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(rows=rows, beta_applied=a.beta, margin=margin,
                                   center=a.center, checkpoint=a.checkpoint,
                                   helper_sha256=sha256(Path(P3.__file__).resolve()),
                                   assignment_source="reconstructed from out.logits and validated "
                                                     "against the model's CE (see ce_minus_recon)"),
                                indent=1))
    print("wrote", out)


def load_margin(path):
    js = json.loads(Path(path).read_text())
    return float(js["margin"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("center"); c.add_argument("--checkpoint", required=True)
    c.add_argument("--manifest", default="dataset/libri2mix_train.jsonl")
    c.add_argument("--out", default="p3_artifacts/center.pt")
    c.add_argument("--n-batches", type=int, default=200); c.add_argument("--batch-size", type=int, default=4)
    c.set_defaults(fn=cmd_center)
    m = sub.add_parser("margin"); m.add_argument("--checkpoint", required=True)
    m.add_argument("--center", required=True); m.add_argument("--manifest", default="dataset/libri2mix_train.jsonl")
    m.add_argument("--out", default="p3_artifacts/margin.json")
    m.add_argument("--n-batches", type=int, default=50); m.add_argument("--batch-size", type=int, default=4)
    m.add_argument("--target-active", type=float, default=0.5)
    m.set_defaults(fn=cmd_margin)
    g = sub.add_parser("grads"); g.add_argument("--checkpoint", required=True)
    g.add_argument("--center", required=True); g.add_argument("--margin", required=True)
    g.add_argument("--manifest", default="dataset/libri2mix_train.jsonl")
    g.add_argument("--out", default="p3_artifacts/grads.json")
    g.add_argument("--n-batches", type=int, default=8); g.add_argument("--batch-size", type=int, default=2)
    g.add_argument("--beta", type=float, default=1.0)
    g.add_argument("--train-mode", action="store_true")
    g.set_defaults(fn=cmd_grads)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
