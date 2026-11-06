# Scripts

Scripts implementing the four-stage paper pipeline (collect → judge → refusal-check → harm-classify) plus calibration tooling. See the repo-root README for the end-to-end commands.

## vLLM Server Launchers

| Script | Description |
|--------|-------------|
| `launch_vllm.sh` | Launch a vLLM server for a local model |
| `launch_hyperclovax_vllm.sh` | Launch HyperCLOVA X vLLM server (specific config) |
| `chain_local_models.sh` | Chain Korean local model collections (Kanana → K-EXAONE → Solar) |

## Smoke Test

| Script | Description |
|--------|-------------|
| `test_public_dataset_smoke.py` | Smoke-test the loader against the public TSV |

## Stage 1 — Response Collection

| Script | Description |
|--------|-------------|
| `collect_model_responses.py` | Collect responses from API models |
| `collect_hyperclovax_responses.py` | Collect responses from HyperCLOVA X (vLLM) |
| `collect_kanana2_responses.py` | Collect responses from Kanana-2 (vLLM) |
| `collect_kexaone_responses.py` | Collect responses from K-EXAONE (vLLM) |
| `collect_solar_responses.py` | Collect responses from Solar-Open (vLLM) |

## Direct-Request Variants

| Script | Description |
|--------|-------------|
| `generate_simplified_direct_requests.py` | Generate the short direct-request variants used in the paper's direct-request analysis (Section 4.5) |

## Stage 2 — Judging

| Script | Description |
|--------|-------------|
| `run_full_judging.py` | Run a single judge model over a directory of response files |
| `run_api_judge_parallel.sh` | Run an API judge (o4-mini, Kimi K2) in parallel via tmux |
| `run_hyperclovax_judge_parallel.sh` | Run the HyperCLOVA X judge in parallel (uses local vLLM) |

## Stage 3 — Refusal Detection

| Script | Description |
|--------|-------------|
| `run_refusal_check.py` | Refusal classifier for adversarial + benign responses |

## Stage 4 — Harm-Dimension Classification

| Script | Description |
|--------|-------------|
| `classify_harm_dimensions.py` | Classify each rubric criterion into the 7 harm dimensions |

## Metaeval Scripts (calibration + cross-variant analysis)

Located in `scripts/metaeval/`:

| Script | Description |
|--------|-------------|
| `run_judge_calibration.py` | Run judge calibration on the 97-task calibration set |
| `run_all_judge_calibration.sh` | Run all judge calibrations |
| `run_all_judges.sh` | Run all judges for calibration |
| `run_korean_judges.sh` | Run Korean model judges |
| `run_all_harm_classifiers.sh` | Run all harm classifiers |
| `run_harm_classifier.py`, `run_harm_classifier_full.py`, `run_harm_classifier_vllm.py` | Harm-dimension classifier runners (API + vLLM variants) |
| `analyze_judge_calibration.py` | Single-judge F1/κ table (Appendix `tab:judge_perf`) |
| `analyze_judge_panel_kappa.py` | Per-variant κ table (Appendix `tab:per_variant_kappa`) |
| `analyze_harm_classifier_calibration.py` | Dimension-classifier κ (κ=0.592 in main text, `tab:dim_classifier_perf`) |
| `analyze_domain_drops.py` | Domain-stratified ΔLing / ΔCtx with bootstrap CIs (paper figure `fig_domain_drops.pdf`) |
| `prepare_judge_calibration_data.py` | Prepare calibration set inputs |
| `sample_meta_eval_dataset.py`, `sample_meta_eval_with_all_variants.py` | Sample the 97-task / 266-task-variant calibration set |
| `dedup_models.py` | De-duplicate model variant entries (used by the `analyze_*` scripts) |

### Korean Model Judge Scripts

Located in `scripts/metaeval/korean_models/`:

| Script | Description |
|--------|-------------|
| `run_hyperclovax_judge.py` | Run HyperCLOVA X as judge |
| `run_kanana2_judge.py` | Run Kanana-2 as judge |
| `run_solar_open_judge.py` | Run Solar-Open as judge |
| `setup_hyperclovax.sh` | Setup script for HyperCLOVA X (clones OmniServe locally) |
| `run_hyperclovax_calibration.sh` | Run HyperCLOVA X calibration |

These scripts assume a local OmniServe checkout. Run `setup_hyperclovax.sh` first or clone OmniServe into `external/OmniServe` before invoking the judge scripts.
