# ASR-Robust Multimodal Intent Recognition — Final Frozen-Encoder Pipeline

This matches the paper exactly: no fine-tuning, no hardware/deployment section.
Target: 14-15 hours max on a single GPU. Fixes the `/tmp` crash from before.

## On your server: what to delete first

You already have `~/garvit/asr_robust_intent` set up with datasets downloaded.
**Do NOT re-download anything** — just fix the code layer. Delete these old files,
they are superseded and will only cause confusion or the same crash as before:

```bash
cd ~/garvit/asr_robust_intent
rm -f scripts/03_train_all.sh
rm -f scripts/05_run_ablations.sh
rm -f scripts/06_run_robustness.sh
rm -f scripts/08_export_and_quantize.sh
rm -f scripts/run_all.sh
rm -f scripts/run_on_pi.py
rm -f scripts/run_on_pi.sh
rm -f src/train.py
rm -f src/ablate_calibration.py
rm -f src/robustness/eval_robustness.py
rm -f src/stats/mcnemar_test.py
rm -rf src/deployment
rm -f src/models/classical_baselines.py   # only if present -- not used in the final approach
```

Also clear stale multiprocessing junk that caused the `OSError: No space left on
device` crash (the pipeline script now does this automatically each run, but do
it once now too):
```bash
rm -rf /tmp/pymp-*
df -h /tmp
```

## What to copy in from this package

Copy every file from this zip into your existing `~/garvit/asr_robust_intent/`,
overwriting anything with the same name. New/updated files:

- `configs/experiment.yaml` — deployment section removed, matches paper
- `scripts/00_setup_env.sh`, `01_download_datasets.sh`, `02_download_whisper.sh` — updated
- `scripts/04_extract_all_features.sh` — extraction (the expensive step)
- `scripts/05_train_all_heads.sh` — fast head training, full 5 seeds
- `scripts/06_run_ablations_head.sh` — calibration + fusion-level ablations
- `scripts/07_run_robustness_head.sh` — WER robustness sweep (Whisper skipped by default for speed)
- `scripts/run_head_pipeline.sh` — **the one command that runs everything**
- `src/data/*.py` — dataset prep (SLURP bug already fixed), audio-length + text-length derivation
- `src/models/fusion.py` — kept ONLY for `AlphaWeights`/`TemperatureScaler`, which `head_fusion.py` depends on
- `src/models/head_fusion.py` — the actual model used, frozen-encoder + lightweight heads
- `src/features/extract_embeddings.py` — one-time frozen embedding extraction (num_workers lowered to 2, safer)
- `src/train_head.py` — fast training over cached embeddings
- `src/ablate_calibration_head.py`, `src/ablate_fusion_level.py`
- `src/robustness/generate_wer_transcripts.py`, `src/robustness/eval_robustness_head.py`
- `src/stats/aggregate_seeds.py`, `src/stats/mcnemar_test_head.py`
- `src/reporting/build_tables.py` — **deployment table removed**, only builds Table II + Table III now

## Since your datasets are already prepared, start here

```bash
conda activate myenv   # or whatever your working env is called
cd ~/garvit/asr_robust_intent
nvidia-smi              # pick a free GPU
```

Run the audio-length check once, and actually look at its output before proceeding
— this is the single biggest lever on your 14-15 hour budget:
```bash
python3 src/data/derive_audio_len.py --config configs/experiment.yaml
```
If it recommends something much shorter than the current `max_audio_seconds`
values in `configs/experiment.yaml` (very likely, since SLURP/FSC utterances are
short commands), update the config file with the recommended numbers before
continuing.

Then run everything with one command, inside `tmux`:
```bash
mkdir -p logs
tmux new -s pipeline
CUDA_VISIBLE_DEVICES=<gpu_number> bash scripts/run_head_pipeline.sh 2>&1 | tee logs/full_run_$(date +%Y%m%d_%H%M%S).log
```
Detach with `Ctrl+B` then `D`. Reattach anytime with `tmux attach -t pipeline`.

## What was fixed vs. the crash you hit

- **`OSError: [Errno 28] No space left on device`** — caused by leftover
  `/tmp/pymp-*` multiprocessing temp files piling up across repeated crashed
  runs of the old fine-tuning script (which used `persistent_workers=True` with
  6 workers, kept alive for 20 epochs at a time). The new pipeline: (a) deletes
  the old script entirely so it can't be re-run by accident, (b) clears `/tmp`
  automatically at the start of every pipeline run, (c) `train_head.py` doesn't
  use multiprocessing workers at all (it just does dict lookups on cached
  tensors — no need), and (d) `extract_embeddings.py`'s worker count is lowered
  from 6 to 2.
- **DistilBERT `AttributeError` during unfreezing** — moot now, since encoders
  are never unfrozen or fine-tuned at all in this approach.

## After it finishes

Check `results/paper_tables.md` — paste directly into the paper's Table II/III.
Anything not yet run shows as `PENDING`, not a fake number.
