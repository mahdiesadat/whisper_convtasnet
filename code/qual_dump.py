#!/usr/bin/env python3
"""qual_dump.py -- per-utterance decode + the four error analyses.

Answers exactly the four questions from the review:
  (1) is the error concentrated in heavy overlap?
  (2) do words of the interferer leak into the target hypothesis?
  (3) do the two branches produce the same content (collapse)?
  (4) is even the low-overlap subset bad (i.e. the problem is not the separator)?

Saves one JSONL row per dev utterance, then prints the bins.
No training, no optimiser, no P3 term (beta=0).
"""
import json, os, sys, itertools, re
from pathlib import Path

sys.path.insert(0, os.environ.get("P3_CODE_ROOT", "/tmp/p3tree"))
sys.path.insert(0, "/home/ubuntu/ws")
import numpy as np, torch, soundfile as sf
from transformers import WhisperProcessor
from model.modeling_whisper_sidecar import WhisperSidecarForConditionalGeneration
from utils.dataset import CustomDataset, _clean_source_path
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
import p3_masks2 as M

W = Path("/home/ubuntu/ws")
CKPT = sys.argv[1] if len(sys.argv) > 1 else str(W / "outputs/p3_pilot_baseline_v1/pilot/checkpoint-best")
MAN = sys.argv[2] if len(sys.argv) > 2 else str(W / "p3_gen2_run/abs_probe_20261111T042249Z_175995/dev_data_abs.jsonl")
OUT = sys.argv[3] if len(sys.argv) > 3 else str(W / "qual_dump_baseline.jsonl")


def words(s):
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def edit(a, b):
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        nd = np.empty_like(d); nd[0] = i
        for j, cb in enumerate(b, 1):
            nd[j] = min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (ca != cb))
        d = nd
    return int(d[-1])


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    return edit(r, h) / max(len(r), 1), len(r)


proc = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe", no_timestamps=True)
model = WhisperSidecarForConditionalGeneration.from_pretrained(
    CKPT, sidecar_loc=1, num_spks=2, soft_prompt_len=4, target_asr=False,
    sidecar_residual=True, sidecar_activate="sigmoid", mask_norm="none",
    sidecar_gate_init=0.1, attn_implementation="sdpa")
model.set_mel_filters(proc.feature_extractor.mel_filters)
model.eval().half().cuda()
coll = DataCollatorSpeechSeq2SeqWithPadding(processor=proc, num_spks=2, soft_prompt_len=4, target_asr=False)
rows_meta = [json.loads(l) for l in open(MAN)]
ds = CustomDataset(data_list_path=MAN, processor=proc, language="en", timestamps=False, num_spks=2)
print("dev utterances:", len(rows_meta), "| samples in dataset:", len(ds), flush=True)

out = open(OUT, "w")
B = 4
for s in range(0, len(ds), B):
    items = [ds[i] for i in range(s, min(s + B, len(ds)))]
    batch = coll(items)
    feats = batch["input_features"].half().cuda()
    with torch.no_grad():
        gen = model.generate(input_features=feats, max_new_tokens=200, num_beams=1)
    hyp = [proc.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]
    # branch b of sample i is at index i*2 + b
    for k, it in enumerate(items):
        row = rows_meta[s + k]
        refs = [str(x["text"]).strip() for x in row["sentences"]]
        # overlap / activity from the clean sources
        cl = batch["clean_audios"][k].numpy()
        cl_len = batch["clean_lengths"][k].numpy().tolist()
        mk = M.build_masks([cl[i] for i in range(2)], cl_len, int(batch["mix_length"][k]), n_frames=1500)
        ov = float(mk["overlap_active"].float().mean())
        act = [float(mk["source_active"][i].float().mean()) for i in (0, 1)]
        h0, h1 = hyp[2 * k], hyp[2 * k + 1]
        c = []
        for perm in itertools.permutations(range(2)):
            w0, n0 = wer(refs[perm[0]], h0); w1, n1 = wer(refs[perm[1]], h1)
            c.append((w0 * n0 + w1 * n1, perm, (w0, w1), (n0, n1)))
        best = min(c, key=lambda x: x[0])
        tot = sum(best[3])
        rec = dict(i=s + k, utt=os.path.basename(row["audio"]["path"]),
                   refs=refs, hyp=[h0, h1], perm=list(best[1]),
                   wer_branch=[round(best[2][0], 4), round(best[2][1], 4)],
                   wer_utt=round(best[0] / max(tot, 1), 4), n_words=tot,
                   overlap_ratio=round(ov, 4), activity=act,
                   ref_len=[len(words(r)) for r in refs],
                   hyp_same=(words(h0) == words(h1)),
                   leak=round(len(set(words(h0)) & (set(words(refs[1 - best[1][0]])) - set(words(refs[best[1][0]]))))
                              / max(len(words(h0)), 1), 4))
        out.write(json.dumps(rec) + "\n")
    out.flush()
    if s % 40 == 0:
        print("  %d/%d" % (s, len(ds)), flush=True)
out.close()

rows = [json.loads(l) for l in open(OUT)]
def wavg(rs):
    n = sum(r["n_words"] for r in rs)
    return 100.0 * sum(r["wer_utt"] * r["n_words"] for r in rs) / max(n, 1)
print("\n=== overall ===")
print("utterances %d | word-weighted WER %.2f%% | macro WER %.2f%%"
      % (len(rows), wavg(rows), 100 * np.mean([r["wer_utt"] for r in rows])))
qs = np.quantile([r["overlap_ratio"] for r in rows], [0.25, 0.5, 0.75])
print("overlap quartile edges:", [round(float(q), 4) for q in qs])
bins = {"Q1 low": [], "Q2": [], "Q3": [], "Q4 high": []}
for r in rows:
    b = 0 if r["overlap_ratio"] <= qs[0] else 1 if r["overlap_ratio"] <= qs[1] else 2 if r["overlap_ratio"] <= qs[2] else 3
    bins[list(bins)[b]].append(r)
print("\n=== (1)+(4) WER by overlap quartile ===")
for k, v in bins.items():
    if v:
        print("  %-8s n=%3d  WER %.2f%%  mean overlap %.3f" % (k, len(v), wavg(v), np.mean([r["overlap_ratio"] for r in v])))
low = [r for r in rows if r["overlap_ratio"] < 0.10]
print("  low-overlap (<10%%): n=%d  WER %.2f%%" % (len(low), wavg(low) if low else float("nan")))
print("\n=== (3) branch collapse ===")
same = [r for r in rows if r["hyp_same"]]
print("  identical hypotheses: %d/%d (%.1f%%) | their WER %.2f%% vs others %.2f%%"
      % (len(same), len(rows), 100 * len(same) / len(rows),
         wavg(same) if same else float("nan"), wavg([r for r in rows if not r["hyp_same"]])))
print("\n=== (2) leakage of interferer words ===")
print("  mean leak rate %.3f | WER(leak>0.2) %.2f%% vs WER(leak<=0.2) %.2f%%"
      % (np.mean([r["leak"] for r in rows]),
         wavg([r for r in rows if r["leak"] > 0.2]) if any(r["leak"] > 0.2 for r in rows) else float("nan"),
         wavg([r for r in rows if r["leak"] <= 0.2])))
print("\nrows saved to", OUT)
