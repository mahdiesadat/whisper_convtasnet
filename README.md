# Sidecar Separation on a Frozen Whisper Encoder

Reproduction and instrumented stress-testing of the *Sidecar* approach — a small trainable
separator injected into a frozen ASR encoder — for 2-speaker overlapped speech
(Libri2Mix), plus a sequence of carefully measured loss and wiring experiments.

## Headline numbers (dev500)

| Experiment | WER |
|---|---|
| Paper wiring: `sep_dec(mask ⊙ sep_enc(h))` (full replacement) | **not trainable** (≈106 %) |
| Our wiring: `h + α · sep_dec(mask ⊙ sep_enc(h))` (gated residual) | **12.47 %** |
| Data effect (251 → 1,172 speakers, same harness) | **−5.82 points** |
| Plain whisper-small on the mixture (no separator) | **100.5 %** |
| Controlled pilot, baseline (3 k steps from scratch, seed 42) | **18.13 %** |
| Same pilot + target/interferer margin, β = 1.0 (seed 42) | **18.76 %** |

## The one mechanical finding that drove the design

For any loss computed on the separator *output*, `∂L/∂α = 0` **exactly** (cosine is
scale-invariant), so such a term cannot move the residual gate. Only losses on the
*stream* `h + α·s` have a non-zero α-gradient. This was **measured**, not assumed
(see `code/p3_loss_r4.py` and the T11 probe in the report).

## Layout

```
code/      loss core (17 unit tests), temporal masks, per-utterance dump, patchers
results/   per-utterance decoder outputs for both pilot arms + launch manifests and logs
REPORT_FOR_PROFESSOR.md   full project report (Persian)
NEXT_STEPS.md             the literal resume plan
```

## Reproduce

```bash
python -m unittest code/test_p3_masks2.py       # mask/length tests
python code/qual_dump.py <checkpoint> <dev.jsonl> out.jsonl
```

## Not claimed

Seed-to-seed variability, statistical significance and long-run behaviour were **not**
estimated: with one seed, the pilot's −0.63 points is neither proven noise nor proven
effect.
