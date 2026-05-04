# Experiments

This directory contains the full experimental pipeline for the paper. Scripts are organized to follow the order they appear in the paper.

## Pipeline overview

```
                    ┌─ extract_features.py ─┐
   eval_sets/  ─→   │ (run model, cache    │  ─→   sae_uncertainty_<model>.pkl
                    │  SAE activations)    │       (per-question features per layer)
                    └──────────────────────┘
                              ↓
                    ┌─ analyze_quadrant.py ─┐  (in scripts/)
                    │ (Mann-Whitney U,     │  ─→   quadrant_analysis.pkl
                    │  identify 3 cats)    │       (pure_unc / pure_inc / both per layer)
                    └──────────────────────┘
                              ↓
            ┌─────────────────┴──────────────────┐
            ↓                                    ↓
  ┌─ individual_suppression  ┐       ┌─ feature_selection.py ─┐
  │  _by_category.py         │       │ (full pipeline for     │
  │  (top-N each, screen)    │       │  "confounded" category)      │
  └──────────────────────────┘       └────────────────────────┘
            ↓
      individual_suppression_by_category.csv  (acc/ent delta per feature)
            ↓
  ┌─ quick_combined_suppression.py ─┐
  │ (combined of beneficial features) │
  └──────────────────────────────────┘
            ↓
      Final paper results (combined suppression on validation)
```

After the main pipeline, several side experiments use the cached SAE features or feature lists:

- **`cross_dataset_suppression.py`** — apply features discovered on one dataset to another
- **`predict_incorrectness.py`** — train classifiers from feature activations
- **`selective_prediction.py`** — abstention via classifier confidence
- **`self_abstain_*.py`** — 5-option "I don't know" experiments
- **`random_suppression_baseline.py`** — random feature control
- **`feature_selection_ablation.py`** — alternative screening criteria

## Core pipeline

### 1. `extract_features.py` — feature discovery

Runs the model on each MCQ question, captures the residual stream at every layer, encodes through the SAE, and caches per-question feature activations.

**Input:** `--eval-set` (JSON), `--model`, `--sae-layers` (e.g. `0-31`)
**Output:** `sae_uncertainty_<model>.pkl` containing per-question records `{indices, values, correct, entropy}` per layer.

```bash
python experiments/extract_features.py \
    --model meta-llama/Llama-3.1-8B \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
    --output-dir results/sae_uncertainty_mmlu_discovery_all_layers/ \
    --sae-layers 0-31
```

### 2. `scripts/analyze_quadrant.py` — quadrant framework

Splits questions into 4 groups (correct×confident, etc.) using entropy percentile, runs Mann-Whitney U test per feature per layer, and identifies three populations: pure uncertainty, pure incorrectness, confounded ("both").

**Input:** the pkl from step 1
**Output:** `quadrant_analysis.pkl` with significance tests per layer + per-layer CSVs

```bash
python scripts/analyze_quadrant.py \
    --pickle results/sae_uncertainty_mmlu_discovery_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
    --output-dir results/sae_quadrant_mmlu_discovery_all_layers_p25/ \
    --entropy-percentile 25
```

### 3a. `individual_suppression_by_category.py` — per-feature screening

For each feature in a category, registers a single-feature `SAESuppressionHook` and re-runs evaluation. Records accuracy and entropy delta per feature.

Used for **pure_uncertainty** and **pure_incorrectness** screening, and for **confounded** on RACE/Gemma where we apply a top-N cap.

**Input:** `quadrant_analysis.pkl`, eval set
**Output:** `individual_suppression_by_category.csv` (one row per feature: layer, feature_idx, acc_delta, ent_delta, ...)

```bash
python experiments/individual_suppression_by_category.py \
    --quadrant-pkl results/sae_quadrant_mmlu_discovery_all_layers_p25/quadrant_analysis.pkl \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
    --output-dir results/sae_individual_suppression_unc_p25/ \
    --category pure_uncertainty \
    --top-n 5
```

### 3b. `feature_selection.py` — full pipeline (Llama confounded)

End-to-end version of the screening + combined suppression pipeline for the **confounded** category on Llama MMLU. Tests all confounded features individually on discovery, screens beneficial, then runs combined suppression on validation in one go.

This produced the headline +1.10% accuracy result for Llama MMLU.

**Input:** `quadrant_analysis.pkl`, discovery set, validation set
**Output:** `both_feature_selection.pkl` + `individual_suppression_selection.csv` + `validation_suppression.csv`

```bash
python experiments/feature_selection.py \
    --model meta-llama/Llama-3.1-8B \
    --quadrant-pkl results/sae_quadrant_mmlu_discovery_all_layers_p25/quadrant_analysis.pkl \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
    --validation-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
    --output-dir results/sae_both_feature_selection_all_layers_p25/
```

### 4. `quick_combined_suppression.py` — combined suppression

Reads a CSV from step 3a (or 3b), selects beneficial features (acc ≥ 0 AND ent < 0), and suppresses all of them simultaneously on a held-out eval set.

**Input:** individual suppression CSV, eval set
**Output:** prints headline numbers (acc delta, ent delta, flips). Used for cross-dataset transfer and category-specific combined evaluations.

```bash
python experiments/quick_combined_suppression.py \
    --individual-csv results/sae_individual_suppression_both_race_p25/individual_suppression_by_category.csv \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
    --label "RACE_TO_MMLU"
```

## Side experiments

### `cross_dataset_suppression.py`

Loads beneficial features from one dataset's selection pkl and applies them to a different dataset to test transfer (e.g., MMLU→ARC, MMLU→RACE).

### `predict_incorrectness.py`

Uses cached SAE feature activations as input to logistic regression (predict correctness, AUROC) and ridge regression (predict entropy, Spearman ρ). Reports per-layer and all-layers metrics by feature category. Demonstrates predictive dissociation.

**Input:** discovery + validation feature pkls, quadrant pkl
**Output:** `prediction_results.csv` (per-layer AUROC and Spearman by feature set)

### `selective_prediction.py`

Trains a logistic regression classifier on SAE features, sweeps confidence thresholds to compute accuracy-coverage tradeoff curves. Compares against entropy-only baseline.

**Input:** discovery + validation feature pkls, quadrant pkl
**Output:** `selective_prediction.csv` and a per-coverage breakdown

### `self_abstain_experiment.py`

Adds a 5th MCQ option ("E. I don't know") and measures how often the model selects it, broken down by whether the original answer was correct or not.

**Input:** eval set
**Output:** `self_abstain_results.csv`

### `self_abstain_feature_analysis.py`

For questions where the model abstained vs answered, compares SAE feature activations between the 4-option and 5-option prompts. Tests whether adding the abstain option changes the activation of uncertainty/incorrectness features.

### `random_suppression_baseline.py`

Control experiment: matches the per-layer feature counts of our method, but selects features **randomly** from the active SAE features (not from quadrant categorization). Runs the same screening + combined suppression pipeline.

**Input:** individual suppression CSV (to match per-layer counts), discovery set, validation set
**Output:** `random_baseline_results.{csv,pkl}`

### `feature_selection_ablation.py`

Ablates the screening criterion. Reuses individual suppression data from `feature_selection.py` and tests three criteria:

1. `acc ≥ 0 AND ent < 0` (paper default)
2. `ent < 0` only
3. `acc ≥ 0` only

Shows that the accuracy gate is essential — `ent < 0` alone selects features that hurt accuracy.

**Input:** `both_feature_selection.pkl` from step 3b
**Output:** `selection_ablation.{csv,pkl}`
