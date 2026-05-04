# SAE Uncertainty Feature Discovery

Code for the paper *LLMs Encode When They're Wrong: Dissociating Uncertainty and Incorrectness via Sparse Autoencoders*.

We use pretrained SAEs to decompose a model's residual stream into interpretable features, then identify which features encode **uncertainty**, **incorrectness**, or **both** — and show they have distinct causal roles.

## Key Findings

We split MCQ questions into 4 groups based on correctness and output entropy:

|                              | Correct  | Incorrect |
|------------------------------|----------|-----------|
| **Confident** (low entropy)  | A        | B         |
| **Uncertain** (high entropy) | C        | D         |

Two comparisons disentangle uncertainty from incorrectness:
- **C vs A** (both correct, different entropy) → **pure uncertainty** features
- **B vs A** (both confident, different correctness) → **pure incorrectness** features
- Features significant in both → **confounded** features

Suppressing each population at the SAE level reveals three distinct causal roles:

1. **Pure uncertainty** features are essential — suppressing them destroys accuracy (-13.4%)
2. **Pure incorrectness** features are epiphenomenal — suppressing them changes nothing (+0.03%)
3. **Confounded** features are causally harmful — suppressing them **improves accuracy (+1.1%)** and **reduces entropy by 75%**

These results replicate across **Llama-3.1-8B** (Llama Scope SAEs, 32K width) and **Gemma-2-9B** (Gemma Scope SAEs, 16K width), and transfer bidirectionally between MMLU and RACE.

## Setup

```bash
git clone https://github.com/HettyPatel/sae-uncertainty.git
cd sae-uncertainty

conda create -n sae-uncertainty python=3.10 -y
conda activate sae-uncertainty
pip install -e .

huggingface-cli login

# Pre-download SAEs (optional, avoids fetch-during-run)
python scripts/download_saes.py --release llama_scope_lxr_8x --id-template "l{layer}r_8x" --layers 0-31
python scripts/download_saes.py --release gemma-scope-9b-pt-res-canonical --id-template "layer_{layer}/width_16k/canonical" --layers 0-41
```

Models configured in [configs/models.yaml](configs/models.yaml):

| Model | SAE Source | Layers | SAE Width |
|-------|-----------|--------|-----------|
| `meta-llama/Llama-3.1-8B` | Llama Scope (`llama_scope_lxr_8x`) | 32 | 32,768 |
| `google/gemma-2-9b` | Gemma Scope (`gemma-scope-9b-pt-res-canonical`) | 42 | 16,384 |

## Pipeline

```
extract_features.py  →  analyze_quadrant.py  →  individual_suppression_by_category.py  →  quick_combined_suppression.py
   (cache features)      (Mann-Whitney U)         (per-feature screening)                  (combined of beneficial)
```

See [experiments/README.md](experiments/README.md) for detailed input/output of each step.

### 1. Feature extraction

```bash
python experiments/extract_features.py \
    --model meta-llama/Llama-3.1-8B \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
    --sae-layers 0-31 \
    --output-dir results/sae_uncertainty_mmlu_discovery_all_layers/
```

Caches per-question SAE feature activations at every layer. The most expensive step.

### 2. Quadrant analysis

```bash
python scripts/analyze_quadrant.py \
    --pickle results/sae_uncertainty_mmlu_discovery_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
    --output-dir results/sae_quadrant_mmlu_discovery_p25/ \
    --entropy-percentile 25
```

Mann-Whitney U test per feature per layer. `--entropy-percentile 25` means the **bottom 25%** of entropies form the *confident* group and the **top 25%** form the *uncertain* group (middle 50% excluded for clean separation).

### 3. Individual feature screening

```bash
python experiments/individual_suppression_by_category.py \
    --model meta-llama/Llama-3.1-8B \
    --quadrant-pkl results/sae_quadrant_mmlu_discovery_p25/quadrant_analysis.pkl \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
    --output-dir results/sae_individual_suppression_unc_p25/ \
    --category pure_uncertainty --top-n 5
```

Run for `pure_uncertainty`, `pure_incorrectness`, `both`. The `--top-n` cap selects features by effect size (omit for confounded on Llama MMLU; we test all 102).

For Llama MMLU confounded specifically, [experiments/feature_selection.py](experiments/feature_selection.py) is an end-to-end version that runs individual screening + combined validation in one go.

### 4. Combined suppression

```bash
python experiments/quick_combined_suppression.py \
    --individual-csv results/sae_individual_suppression_unc_p25/individual_suppression_by_category.csv \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
    --label "PURE_UNCERTAINTY"
```

Selects features with `acc_delta ≥ 0` AND `ent_delta < 0` from the individual CSV, then suppresses all of them simultaneously on the held-out validation set.

## Side Experiments

### Cross-dataset transfer

Apply features discovered on one dataset to another:

```bash
python experiments/quick_combined_suppression.py \
    --individual-csv results/sae_individual_suppression_both_p25/individual_suppression_by_category.csv \
    --eval-set data/eval_sets/eval_set_mcq_arc_challenge_2577.json \
    --label "MMLU_TO_ARC"
```

### Random feature baseline

Match the per-layer feature counts of our method but pick features randomly from active SAE features. Same screening + combined pipeline; only difference is feature selection.

```bash
python experiments/random_suppression_baseline.py \
    --model meta-llama/Llama-3.1-8B \
    --individual-csv results/sae_both_feature_selection_p25/individual_suppression_selection.csv \
    --discovery-pkl results/sae_uncertainty_mmlu_discovery_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
    --discovery-eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
    --output-dir results/sae_random_baseline_mmlu_p25/
```

### Predictive dissociation

Train classifiers on cached SAE features to predict correctness (logistic regression) and entropy (ridge regression):

```bash
python experiments/predict_incorrectness.py \
    --discovery-pkl results/sae_uncertainty_mmlu_discovery_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
    --validation-pkl results/sae_uncertainty_mmlu_validation_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
    --quadrant-pkl results/sae_quadrant_mmlu_discovery_p25/quadrant_analysis.pkl \
    --output-dir results/sae_predict_incorrectness_p25/
```

### Selective prediction

Use classifier confidence to abstain on low-confidence questions:

```bash
python experiments/selective_prediction.py \
    --discovery-pkl results/sae_uncertainty_mmlu_discovery_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
    --validation-pkl results/sae_uncertainty_mmlu_validation_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
    --quadrant-pkl results/sae_quadrant_mmlu_discovery_p25/quadrant_analysis.pkl \
    --output-dir results/sae_selective_prediction_p25/
```

### Self-abstention

Add "E. I don't know" as a 5th MCQ option and measure how the model's choice and feature activations change:

```bash
python experiments/self_abstain_experiment.py \
    --model meta-llama/Llama-3.1-8B \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
    --output-dir results/sae_self_abstain_experiment/
```

### Selection criterion ablation

Compare `acc + ent`, `acc only`, `ent only` screening criteria using already-collected individual suppression data:

```bash
python experiments/feature_selection_ablation.py \
    --individual-csv results/sae_both_feature_selection_p25/individual_suppression_selection.csv \
    --validation-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
    --output-dir results/sae_feature_selection_ablation/
```

## Headline Numbers (Llama-3.1-8B, MMLU validation, 7,021 samples)

Baseline: **61.91%** accuracy, **0.824** mean entropy.

**Combined suppression of beneficial features:**

| Category | Tested | Beneficial | Acc Δ | Ent Δ |
|----------|--------|-----------|-------|-------|
| Confounded | 102 | 55 | **+1.10%** | **-0.619** |
| Pure uncertainty | 160 | 63 | +0.21% | -0.382 |
| Pure incorrectness | 137 | 52 | +0.03% | ~0 |
| Random baseline | 102 | 35 | +0.04% | -0.005 |

**Cross-dataset transfer (confounded features):**

| Discovery → Eval | Features | Acc Δ |
|------------------|----------|-------|
| MMLU → ARC | 55 | +0.81% |
| MMLU → RACE | 55 | +0.70% |
| RACE → MMLU | 30 | +0.75% |

Gemma replication and full per-layer breakdowns in [results/feature_manifest/](results/feature_manifest/).

## Suppression Mechanism

The `SAESuppressionHook` in [src/sae.py](src/sae.py) performs a surgical intervention on the residual stream:

1. At the target layer, intercept the hidden state at the last token
2. Encode → SAE features
3. Zero out target feature activations
4. Decode original and modified separately
5. Add `(modified_decoded - original_decoded)` as a delta on top of the original hidden state

This isolates the targeted features' contribution while preserving all other information (the SAE's reconstruction error is identical in both branches and cancels out).

## Project Structure

```
sae-uncertainty/
├── configs/models.yaml             # Model + SAE configuration
├── data/eval_sets/                 # MCQ evaluation JSONs (see folder README)
├── experiments/                    # Pipeline scripts (see folder README)
├── scripts/                        # Setup, analysis, and plotting utilities
├── src/                            # Model loading, evaluation, SAE hooks
├── figures/                        # Paper figures
├── results/feature_manifest/       # Curated tested + beneficial features (CSV)
└── pyproject.toml
```

Random seed is 42 throughout. All experiments use the entropy-percentile-25 threshold for the quadrant split.
