# راه‌اندازی مخزن GitHub

> **صادقانه:** من به حساب GitHub شما دسترسی ندارم (نه توکن، نه SSH key، و نه اجازهٔ شبکه به github.com). پس **نمی‌توانم چیزی push کنم**. کاری که می‌کنم: یک ساختار آمادهٔ repo + فایل‌ها را می‌دهم که با ۵ دستور خودتان بالا بیاورید.

---

## گام ۱ — ساخت مخزن خالی

1. به <https://github.com/new> بروید
2. نام: `whisper-sidecar-sep` (یا هر نامی)
3. **Public** یا **Private** — برای گزارش به استاد، Private کافی است
4. **هیچ** گزینه‌ای را تیک نزنید (نه README، نه .gitignore، نه license) — چون خودمان اضافه می‌کنیم
5. Create repository

## گام ۲ — ساخت ساختار پوشه‌ها

پیشنهاد ساختار (کد + گزارش، بدون وزن‌های مدل):

```
whisper-sidecar-sep/
├── README.md                  ← متن پایین
├── REPORT_FOR_PROFESSOR.md    ← گزارش کامل
├── NEXT_STEPS.md
├── .gitignore
├── code/
│   ├── p3_loss.py             ← هستهٔ لاس (۱۷ تست)
│   ├── p3_masks2.py
│   ├── p3_training.py
│   ├── qual_dump.py
│   ├── p3_artifacts_r2.py
│   ├── run_p3_bounded_pilot.py
│   └── patch_*.py
├── results/
│   ├── qual_dump_baseline.jsonl
│   ├── qual_dump_p3.jsonl
│   ├── pilot_baseline_v1/     ← launch_manifest + finetune_config + train.log
│   └── pilot_p3_v1/
└── docs/
    └── figures/               ← ۵ شکل تحلیل
```

> ⚠️ **وزن‌های مدل را push نکنید** (۹۵۵ MB). آن‌ها روی دیسک شما هستند و برای GitHub مناسب نیستند. اگر لازم شد: GitHub Releases یا Git LFS.

## گام ۳ — دستورها (در پوشهٔ پروژه)

```bash
cd "C:\Users\Mahdie Sadat\Desktop\mainproject"
git init
git add README.md REPORT_FOR_PROFESSOR.md NEXT_STEPS.md .gitignore code results docs
git commit -m "Sidecar reproduction + P3 experiments: report, code, per-utterance results"
git branch -M main
git remote add origin https://github.com/<USERNAME>/whisper-sidecar-sep.git
git push -u origin main
```

> اگر برای اولین بار push می‌کنید، GitHub یک **Personal Access Token** می‌خواهد (Settings → Developer settings → Tokens → Fine-grained). رمز عبور ساده دیگر کار نمی‌کند.

## گام ۴ — بازرسی

در صفحهٔ repo: باید `README.md` بنفش‌رنگ (رندرشده) بالای صفحه دیده شود و `results/qual_dump_*.jsonl` در پوشه.

---

# `.gitignore` (این را در ریشه بسازید)

```gitignore
# weights and checkpoints
*.safetensors
*.bin
*.pt
*.ckpt
*.pth
checkpoint-*/
pilot/

# datasets and audio
dataset/
raw/
*.wav
*.flac

# caches
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/
.DS_Store
Thumbs.db

# logs with machine paths (keep the ones under results/)
outputs/
logs/
```

---

# `README.md` (این را در ریشه بسازید)

```markdown
# Sidecar Separation on a Frozen Whisper Encoder — reproduction + analysis

Reproducing and stress-testing the *Sidecar* approach (a small trainable separator
injected into a frozen ASR encoder) for 2-speaker overlapped speech, plus a sequence of
carefully instrumented loss and wiring experiments.

## What is here

| Path | Content |
|---|---|
| `REPORT_FOR_PROFESSOR.md` | Full project report (Persian): why the paper's wiring was untrainable here, what we changed, every loss tried, the diagnostics, the pilot result, the next plan |
| `code/p3_loss.py` | The target-vs-interferer margin loss + the norm guard, with 17 unit tests |
| `code/p3_masks2.py` | Temporal masks: valid lengths captured *before* padding, union for the centre, overlap for the loss |
| `code/qual_dump.py` | Per-utterance decode + the four error analyses (overlap bins, leakage, branch collapse, low-overlap subset) |
| `results/` | Per-utterance decoder outputs for both pilot arms + launch manifests and logs |
| `NEXT_STEPS.md` | The literal resume plan |

## Headline numbers

| Experiment | WER (dev500) |
|---|---|
| Paper wiring (`sep_dec(mask⊙sep_enc(h))`, replacement) | **not trainable** (≈106 %) |
| Our wiring (`h + α·sep_dec(...)`, gated residual) | **12.47 %** |
| Data effect (251 → 1,172 speakers, same harness) | **−5.82 points** |
| Plain whisper-small on the mixture (no separator) | **100.5 %** |
| Controlled pilot, baseline (3 k steps, from scratch) | **18.13 %** |
| Same pilot + target/interferer margin (β=1.0) | **18.76 %** (−0.63, i.e. worse) |

## The one mechanical finding that drove the design

For any loss computed on the separator *output*, `∂L/∂α = 0` exactly (cosine is
scale-invariant), so such a term **cannot** move the residual gate. Only losses on the
*stream* `h + α·s` have a non-zero α-gradient. This was measured (T11), not assumed.

## Reproduce

```bash
python -m unittest code/test_p3_loss.py      # 17 tests
python code/qual_dump.py <checkpoint> <dev_manifest.jsonl> out.jsonl
```

## Not claimed

Seed-to-seed variability, statistical significance, and long-run behaviour were **not**
estimated: with one seed the pilot's −0.63 points is neither proven noise nor proven
effect.

## License

Research code, provided as-is for review.
```
