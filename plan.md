# ARC Prize 2026 - ARC-AGI-2: Study & Execution Plan

- **Status:** Active
- **Created:** 2026-09-16
- **Owner:** johnsonhk88
- **Competition deadline:** 2026-11-02 (winners announced 2026-12-04)
- **Repo:** https://github.com/johnsonhk88/kaggle-ARC-Prize-2026-ARC-AGI-2

This file is the plan of record. Update checkboxes and the progress log as work proceeds.

---

## 1. Goals & Success Criteria

1. **Baseline submission (primary).** Take the existing `qwen3_4b_grids15_sft139` TTT + turbo-DFS pipeline, fix its reliability problems, and submit a notebook to the Kaggle competition.
   - Target: public LB in the **~30-34%** band.
   - Reference numbers: best public score attached to this model on the 2026 competition = **33.89**; identical-config reruns = **26.9 / 29.7 / 31.4 / 31.8 / 32.2** (avg 30.4); NVARC official 2025 = **27.64% public**, 24% private.
   - The "32.00 average" figure the user mentioned is not documented anywhere public; treat 32 as a target to reproduce/beat, not a published constant.
2. **Study (documentation).** Produce `docs/arc-prize-2026-study.md` covering the three requested topics:
   1. How to run inference with this model and submit to the competition.
   2. How to fine-tune this model / port the recipe to new models.
   3. How the competition data and the training datasets fit together.
3. **Fine-tuning (secondary).** LoRA fine-tune newer models for ARC-AGI-2 and evaluate with the same test-time-training + DFS harness:
   - Ladder: `Qwen/Qwen3.5-4B` -> `google/gemma-4-E4B-it` -> a larger ~8-14B model (verify availability) on Colab Pro.
   - Evaluate pass@2 on 100 held-out train tasks + the 120 public eval tasks.
4. **Improved submission (stretch).** If a fine-tuned model beats the baseline on evaluation, run TTT + DFS with it and submit before 2026-11-02.

## 2. Hardware & Where Work Happens

| Environment | Hardware | Use |
| --- | --- | --- |
| Local workstation | RTX PRO 4500 Blackwell, 32GB VRAM, 94GB RAM, 1.6TB free | Data prep, small/full LoRA runs up to ~12B (QLoRA for the largest), local harness + smoke tests |
| Google Colab Pro | A100 40GB or RTX PRO 6000 (96GB) | Larger bf16 LoRA runs, longer training |
| Kaggle | L4x4 (4x L4 24GB each) | Competition inference only (12h limit, no internet, 30h/week quota) |

Notes:
- Local GPU is Blackwell (sm_120): needs PyTorch cu128+ wheels; Unsloth compatibility must be verified (fallback: plain PEFT/TRL LoRA without Unsloth).
- Kaggle L4 notebooks consume GPU quota at 2x the T4 rate; plan few, long runs.
- No `~/.kaggle/kaggle.json` exists yet -> blocking for Kaggle data/model downloads (user action).

## 3. Verified Facts (research summary)

### 3.1 Competition
- Code competition, CPU/GPU notebook <= 12h, no internet, external public data and pre-trained models allowed, submission file `submission.json`.
- Scoring: for every test input, 2 attempts; if either matches the ground truth exactly the output scores 1; final score is the average over all task test outputs.
- Data page files: `arc-agi_training_challenges.json` (1,000 tasks), `arc-agi_training_solutions.json`, `arc-agi_evaluation_challenges.json` (120 tasks), `arc-agi_evaluation_solutions.json`, `arc-agi_test_challenges.json` (240-task placeholder; replaced by unseen tasks on rerun), `sample_submission.json`.
- Hidden scoring uses 240 unseen tasks; public LB is one half, final/private LB the other half.
- Prize eligibility requires open-sourcing all code/methods (permissive license such as CC0/MIT-0; third-party code at least Apache-2.0/GPLv3). Internet is not available during evaluation.

### 3.2 Baseline model: `qwen3_4b_grids15_sft139`
- Kaggle: https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139 (Apache-2.0, bfloat16, ~7.27GB, 37 votes).
- HF mirror: https://huggingface.co/Grusmannen97/qwen3_4b_grids15_sft139-transformers-bfloat16-v1 (MIT, 7.27GB, empty model card).
- Base: `Qwen/Qwen3-4B-Thinking-2507` with tokenizer/embedding cut to a **16-token vocab** (digits 0-9, newline, user, assistant, 3 specials). `vocab_size: 16`, 3.63B params in BF16.
- Training: NVARC (1st place ARC Prize 2025) full fine-tune via NVIDIA NeMo-RL/Megatron on `grids_v15`: 1 epoch, 12,716 steps, global batch 256, lr 1e-4, sequence packing to 256k tokens, 4 nodes x 8 H100 for 27h. Dataset: 3.2M augmented samples.
- Inference (official): per-puzzle LoRA test-time training + thresholded turbo DFS.
  - TTT: `r=256`, `alpha=32`, `use_rslora=True`, dropout 0, targets `q/k/v/o/gate/up/down_proj + embed_tokens + lm_head`, bf16 (not 4-bit), `max_seq_length=8192`, 1 epoch, lr 5e-5, cosine, 128 augmented views (16 color x 8 dihedral).
  - Decode: no sampling; DFS over 16 tokens with cumulative NLL threshold `-log(0.2)`, max 540s/DFS call; 16 augmented views per test input; candidate ranking by product-of-probabilities and KGMoN-style vote; top-2 unique grids become `attempt_1`/`attempt_2`.
  - Grid format: `user\n<digits row>\n...assistant\n`, one token per cell with the 16-token vocab.
- Published wins/paper/code: https://github.com/1ytic/NVARC (paper `nvarc_2025.pdf`).

### 3.3 Sample notebooks in this repo
`src/sample-code/qwen3_4b_grids15_sft139/`:
- `failed-in-aimo.ipynb` and `reproduce-nvarc-2025-results.ipynb`: byte-identical base code.
- `arc-agi2-original-kg.ipynb`: base + deterministic augmentation seed + robust model-dir resolution (run 2026-06-18, local 2.5/4).
- `arc-agi-enhanced-solution.ipynb`: refactor + ensemble ranker (run 2026-04-15, local 2.5/4).

Defects to fix in the hardened notebook:
- Queue draining race (`while not queue.empty()` on an `mp.Manager().Queue()`).
- Non-deterministic verification seed (`hash(bk) % 1024**2`) in 3 of 4 notebooks (only `arc-agi2-original-kg` fixes it).
- Fallback prediction for unprocessed tasks is `[[0]]` (1x1 zero grid) - wastes the 2 attempts.
- No per-worker time budgeting across the 240 tasks; a slow task can starve others.
- `fill_submission` key parsing via `k.split("_")` (fixed only in the enhanced notebook).
- Validation only checks grid shape 1..30, not the expected output shape.
- TTT adapters are never saved; only decoded hypotheses are pickled.

### 3.4 New fine-tuning targets
- `Qwen/Qwen3.5-4B`: 4.66B total, Apache-2.0, 262k context, vision-capable, thinking mode. Unsloth: bf16 LoRA ~10GB VRAM; QLoRA is explicitly discouraged for Qwen3.5.
- `google/gemma-4-E4B-it`: 4.5B effective (~8B with embeddings), Apache-2.0 (Gemma 4 license text), 128k context, multimodal. Unsloth: LoRA ~17GB, QLoRA ~10GB.
- Larger options to verify at Phase 4: Qwen3.5 ~8-14B, Gemma 4 ~12B variants (availability/licensing check first).
- Stack: Unsloth + PEFT + TRL `SFTTrainer`; sequence packing; `train_on_responses_only`-style completion-only loss; bf16 LoRA preferred.

### 3.5 Datasets
- Competition data (1,000 train + 120 eval) - Apache-2.0, canonical source.
- **NVARC Augmented Puzzles** (approved for download): https://www.kaggle.com/datasets/sorokin/nvarc-augmented-puzzles - 11.97GB, 3.2M samples, 42 files, Arrow format, 7 subsets (`arc2_training`, `arc2_evaluation6`, `concept`, `mini`, `nvarc_training`, `nvarc_full`, `rearc`), license "Unknown".
- NVARC Synthetic Puzzles: https://www.kaggle.com/datasets/sorokin/nvarc-synthetic-puzzles - 5.79GB, 103k synthetic puzzles.
- NVARC Artifacts Puzzles: https://www.kaggle.com/datasets/sorokin/nvarc-artifacts-puzzles - 47.5GB (optional).
- Public generators: ARC-GEN (https://github.com/google/ARC-GEN, Apache-2.0, 100K dataset on Kaggle), RE-ARC (https://github.com/michaelhodel/re-arc, MIT), ConceptARC, MINI-ARC.
- Other SFT-ready: `nvidia/Nemotron-SFT-ARC-AGI-v1`, `mertaylin/arc-agi-transduction100k`, `stephannef/arcgen-sft-2k-whole-family-v1`.
- License caution: NVARC datasets are "Unknown" license; fine for local experiments/benchmarks, but prize eligibility needs open-source code and care with data provenance.

## 4. Decisions

| ID | Decision | Status |
| --- | --- | --- |
| D1 | Train locally + on Colab Pro; Kaggle reserved for competition inference/submission | Resolved |
| D2 | Model size cap raised: ~12B parameter models allowed (QLoRA where VRAM-constrained) | Resolved |
| D3 | Port NVARC's 16-token vocab approach to new models (lossless row selection; validate in a spike; stock-tokenizer + constrained decoding as fallback) | Resolved (validate in P3) |
| D4 | Download NVARC ready-made datasets (augmented + synthetic puzzles) and use them as the primary training data, with competition/ARC-GEN/RE-ARC data as license-clean supplements | Resolved |
| D5 | Baseline first, then fine-tuning | Resolved |
| D6 | Runnable code lives in a **single Jupyter notebook**: `src/notebooks/arc2_ttt_dfs_solver.ipynb` (Kaggle/Colab/local). The standalone `src/arc/*.py` modules and the `src/tools/build_notebook.py` generator were removed after consolidation (2026-09-17) | Resolved |

## 5. Phases & Tasks

Use `[ ]` / `[x]` and append a dated note when a task completes.

### Phase 0 - Environment & data setup
- [~] P0.1 Create Python venv and install the training stack (torch cu128+, transformers, peft, trl, unsloth, datasets, accelerate, jupyter, kaggle, kagglehub, huggingface_hub).
  - [x] 2026-09-16: `.venv` created; `torch==2.11.0+cu128` installed and verified (CUDA available, sm_120 in arch list).
  - [x] 2026-09-16: `requirements.txt` written at repo root; remaining installs delegated to the user.
- [ ] P0.2 **User action:** create Kaggle API token -> save as `~/.kaggle/kaggle.json` (or run `kaggle login`), and accept the competition rules on the Kaggle website.
- [~] P0.3 Download competition data -> `data/competition/`.
  - [x] 2026-09-16: public ARC-AGI-2 repo cloned to `data/ARC-AGI-2` (1000 train + 120 eval, Apache-2.0) and combined Kaggle-style files built to `data/combined/` (unblocks local work without Kaggle auth). Kaggle competition download still optional.
- [ ] P0.4 Download `sorokin/nvarc-augmented-puzzles` (11.97GB) and `sorokin/nvarc-synthetic-puzzles` (5.79GB) -> `data/nvarc/`.
- [x] P0.5 Download the model: HF mirror (7.27GB) -> `models/qwen3_4b_grids15_sft139/` (both shards verified byte-identical to HF).
- [x] P0.6 Add `.gitignore` for `data/`, `models/`, `runs/`, `.venv/`, caches.
- [x] P0.7 GPU smoke test: model loads in 1.2s (3.63B params, 7.27GB VRAM); the custom tokenizer MUST be `PreTrainedTokenizerFast(tokenizer_file=tokenizer.json)` (`AutoTokenizer` loads a wrong variant that drops `user`/`assistant`); 2-step LoRA TTT trained successfully.

### Phase 1 - Local harness + study
- [x] P1.1 Extract the notebook modules and fixes into a single self-contained notebook `src/notebooks/arc2_ttt_dfs_solver.ipynb` (data build, model fetch, TTT, DFS decode, ranking, submission, validation). Standalone `.py` modules removed per D6.
- [~] P1.2 Local smoke test; expect ~2.5-3.0/4 on the 4 hardcoded eval tasks; record per-task TTT/decode times.
  - 2026-09-17: dry-run of the notebook passes with 0 errors. Background 1-task run (`0934a4d8`, `ARC_GRAD_CKPT=1`) completed end-to-end: 13 candidate files + `submission.json`; validation **0/1** (correct grid absent from candidates; local plain-transformers path differs from the Unsloth path → directional only). User runs remaining task validation.
- [ ] P1.3 Write `docs/arc-prize-2026-study.md` (inference pipeline; fine-tuning new models; data/dataset guide) with source links.
- [ ] P1.4 Update `README.md` with repo structure and quick start.

### Phase 2 - Hardened Kaggle notebook + baseline submission
- [ ] P2.1 Build `src/kaggle/arc2_sft139_ttt_inference.ipynb`: deterministic seeds, sentinel-based queue, model-path resolution, per-worker time budgeting (`remaining/remaining_tasks`), best-train-output fallback instead of `[[0]]`, output-shape-aware fallback, submission format check against `sample_submission.json`.
- [ ] P2.2 Dry-run the notebook logic locally on a small subset.
- [ ] P2.3 Upload/run on Kaggle with competition data + model attached (L4x4, <= 12h); submit; record public LB.
- [ ] P2.4 Log result and observations in the progress log.

### Phase 3 - Dataset builder
- [ ] P3.1 Spike: inspect Qwen3.5 / Gemma 4 tokenizers; verify digit single-token behavior and 16-token vocab-cut feasibility (row selection mapping).
- [ ] P3.2 Build `src/finetune/data/build_dataset.py`: serialize tasks as multi-turn grid dialogues; 8 dihedral x color-permutation augments; leave-one-out query format; splits = 100 held-out train tasks, 120 public eval untouched; output JSONL/Arrow.
- [ ] P3.3 Tokenize + validate (round-trip grid parse), dedupe against eval, report stats.
- [ ] P3.4 Baseline dataset variant from NVARC augmented/synthetic puzzles for comparison.

### Phase 4 - Fine-tuning runs
- [ ] P4.1 Qwen3.5-4B LoRA config (r sweep 16/32/64, all linear + embed/lm_head, bf16, seq 8192, packing, lr 1e-4..2e-4, 1-3 epochs); train locally, scale on Colab.
- [ ] P4.2 Eval pass@2 (no TTT) on held-out 100 + public eval; compare against the SFT139 baseline.
- [ ] P4.3 Gemma 4 E4B same protocol (LoRA or QLoRA).
- [ ] P4.4 Larger ~8-14B candidate (verify availability/license; Colab PRO 6000/A100).
- [ ] P4.5 TTT + DFS inference with the best fine-tuned model; compare to baseline TTT results.

### Phase 5 - Improved submission (stretch)
- [ ] P5.1 Port the best fine-tuned model into the hardened Kaggle notebook; submit before 2026-11-02.
- [ ] P5.2 Optional improvements if time allows: more TTT views, adaptive budgets, output-shape predictor, alternative ranking algorithms.

## 6. Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Run-to-run LB variance (26.9-32.2 observed) | Deterministic seeds; multiple submissions; report best and median |
| 240 tasks in 12h on 4x L4 is tight | Per-worker time budget, work stealing, adaptive view counts, profiling in P1.2 |
| Unsloth incompatible with Blackwell sm_120 | Fall back to plain PEFT/TRL LoRA; validate in P0.7 |
| 16-token vocab cut fails on Qwen3.5/Gemma 4 tokenizers | Spike in P3.1; fall back to stock tokenizer + constrained decoding |
| NVARC data license "Unknown" (prize eligibility) | Prefer license-clean sources for any prize-eligible artifact; use NVARC data for internal experiments |
| Only ~6.5 weeks to the deadline | Baseline submission lands early (P2); fine-tuning is incremental |

## 7. Open Questions

- Colab Pro GPU tier available (A100 40GB vs RTX PRO 6000 96GB)? Determines max model size and bf16 vs QLoRA.
- Confirm the Kaggle submission will be made under this user's account (needed for P2.3).
- Which exact larger model to target in P4.4 (verify what exists at that time).

## 8. References

- Competition overview: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview
- Competition data: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data
- ARC Prize 2026: https://arcprize.org/competitions/2026 ; ARC-AGI-2: https://arcprize.org/arc-agi/2
- ARC-AGI-1/2 technical guide: https://arcprize.org/guide/1
- NVARC code/paper: https://github.com/1ytic/NVARC
- Baseline model: https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139 ; https://huggingface.co/Grusmannen97/qwen3_4b_grids15_sft139-transformers-bfloat16-v1
- Official submission notebook: https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue
- Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B ; Gemma 4 E4B: https://huggingface.co/google/gemma-4-E4B-it
- Unsloth docs: https://unsloth.ai/docs/models/qwen3.5/fine-tune.md ; https://unsloth.ai/docs/models/gemma-4/train.md
- DARC/RE-ARC: https://github.com/michaelhodel/re-arc ; ARC-GEN: https://github.com/google/ARC-GEN

## 9. Progress Log

| Date | Phase | Note |
| --- | --- | --- |
| 2026-09-16 | - | Plan approved and written. Research complete: model lineage, notebooks, datasets, fine-tuning stack. |
| 2026-09-16 | P0 | venv + torch 2.11.0+cu128 verified (sm_120 OK). `requirements.txt` written; user installs remaining deps. |
| 2026-09-16 | P0.5 | HF mirror of the baseline model downloading in background -> `models/qwen3_4b_grids15_sft139/` (log: `download.log`). |
| 2026-09-17 | P0 | Model verified complete. Tokenizer gotcha found + fixed (fast tokenizer from `tokenizer.json`). venv stack installed. |
| 2026-09-17 | P1 | Single-notebook architecture (D6): `src/notebooks/arc2_ttt_dfs_solver.ipynb`; dry-run passes with 0 errors; 1-task background run completed (TTT+DFS+submission). Standalone `.py` modules removed. |
