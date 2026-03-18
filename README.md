# SAE Uncertainty Feature Discovery

Discovering and manipulating internal uncertainty representations in LLMs using Sparse Autoencoders (SAEs).

We use pretrained SAEs to decompose a model's residual stream into interpretable features, then identify which features encode **uncertainty**, **incorrectness**, or **both** — and show they have distinct causal roles.

## Key Idea

We split MCQ questions into 4 groups based on correctness and output entropy:

|                        | Correct              | Incorrect            |
|------------------------|----------------------|----------------------|
| **Confident** (low entropy)  | Group A              | Group B              |
| **Uncertain** (high entropy) | Group C              | Group D              |

Two comparisons disentangle uncertainty from incorrectness:
- **C vs A** (both correct, different entropy) → **pure uncertainty** features
- **B vs A** (both confident, different correctness) → **pure incorrectness** features
- Features significant in **both** → **confounded ("both")** features

Suppression experiments reveal three causal roles:
1. **Pure uncertainty** features are essential — suppressing them destroys accuracy
2. **Pure incorrectness** features are epiphenomenal — suppressing them changes nothing
3. **"Both" features** are causally harmful — suppressing them **improves accuracy** and **halves entropy**

## Setup

```bash
# Clone
git clone https://github.com/HettyPatel/sae-uncertainty.git
cd sae-uncertainty

# Create environment
conda create -n sae-uncertainty python=3.10 -y
conda activate sae-uncertainty

# Install package (editable mode)
pip install -e .

# Login to HuggingFace (needed to download SAEs and models)
huggingface-cli login

# Pre-download SAEs (recommended, avoids flaky downloads during long runs)
python scripts/download_saes.py
```

## Supported Models

Configured in `configs/models.yaml`:

| Model | SAE Source | Status |
|-------|-----------|--------|
| `meta-llama/Llama-3.1-8B` | Llama Scope (8x) | Primary, fully tested |
| `google/gemma-2-9b` | Gemma Scope | Config ready |
| `google/gemma-2-9b-it` | Gemma Scope | Config ready |
| `Qwen/Qwen2.5-7B-Instruct` | Community SAEs | Config ready |

## Pipeline

The full pipeline has 6 steps. Steps 1-2 are data prep (CPU only). Step 3 is GPU-intensive. Steps 4-6 require GPU for model inference.

### Step 1: Create evaluation set

```bash
python scripts/create_mmlu_eval_set.py --split test --samples 14042
```

Creates `data/eval_sets/eval_set_mcq_mmlu_test_14042.json`. You can also use existing eval sets in `data/eval_sets/` (ARC-Challenge, BoolQ, NQ).

### Step 2: Split into discovery / validation

```bash
python scripts/split_eval_set.py \
    --input data/eval_sets/eval_set_mcq_mmlu_test_14042.json
```

Produces `*_discovery.json` and `*_validation.json` (50/50 split). The discovery set is used for feature identification and selection; the validation set is held out for final evaluation.

### Step 3: Extract SAE features

```bash
python experiments/extract_features.py \
    --model meta-llama/Llama-3.1-8B \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
    --sae-layers 0-31 \
    --output-dir results/extract_features_mmlu_discovery/
```

This is the most expensive step (GPU-intensive). It:
1. Runs inference on all questions, caching residual stream activations per layer
2. Loads each SAE one at a time, encodes activations, stores sparse feature representations
3. Computes differential statistics (Mann-Whitney U) between correct/incorrect groups

**Outputs:** `sae_uncertainty_Llama-3.1-8B.pkl` (large, contains per-question SAE features), `differential_features_*.csv`, `entropy_features_*.csv`

For long runs, use nohup:
```bash
mkdir -p logs
nohup python experiments/extract_features.py \
    --model meta-llama/Llama-3.1-8B \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
    --sae-layers 0-31 \
    --output-dir results/extract_features_mmlu_discovery/ \
    > logs/extract_features.log 2>&1 &

tail -f logs/extract_features.log
```

**Layer selection:** Use `--sae-layers 0-31` for all layers, or a subset like `--sae-layers 16,20,24,28,31`. If omitted, uses defaults from `configs/models.yaml`.

### Step 4: Quadrant analysis

```bash
python scripts/analyze_quadrant.py \
    --pickle results/extract_features_mmlu_discovery/sae_uncertainty_Llama-3.1-8B.pkl \
    --output-dir results/quadrant_mmlu_discovery/ \
    --entropy-percentile 25
```

CPU only. Classifies features into pure_uncertainty, pure_incorrectness, and both categories.

`--entropy-percentile 25` means bottom 25% entropy = "confident", top 25% = "uncertain" (middle 50% excluded from grouping). Omit for a median split.

**Outputs:** `quadrant_analysis.pkl`, `quadrant_summary.csv`, per-layer CSVs

### Step 5: Suppression by category

```bash
python experiments/suppress_by_category.py \
    --model meta-llama/Llama-3.1-8B \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
    --quadrant-pkl results/quadrant_mmlu_discovery/quadrant_analysis.pkl \
    --output-dir results/suppression_by_category/
```

Suppresses top-N features from each category (pure_uncertainty, pure_incorrectness, both) per layer, then all layers combined. Measures accuracy and entropy changes.

**Outputs:** `suppression_by_category.csv`, `suppression_by_category.pkl`

### Step 6: Feature selection (validation)

```bash
python experiments/feature_selection.py \
    --model meta-llama/Llama-3.1-8B \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \
    --validation-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
    --quadrant-pkl results/quadrant_mmlu_discovery/quadrant_analysis.pkl \
    --output-dir results/feature_selection/
```

Tests each "both" feature individually on the discovery set, selects ones where accuracy >= baseline AND entropy < baseline, then evaluates the selected set on the held-out validation data.

**Outputs:** `individual_suppression_selection.csv`, `validation_suppression.csv`, `both_feature_selection.pkl`

## Project Structure

```
sae-uncertainty/
├── configs/
│   └── models.yaml              # Model + SAE configurations
├── data/
│   └── eval_sets/               # MCQ evaluation JSONs (MMLU, ARC, BoolQ, NQ)
├── experiments/
│   ├── extract_features.py      # Step 3: inference + SAE encoding + differential stats
│   ├── suppress_by_category.py  # Step 5: suppress features by category, measure impact
│   └── feature_selection.py     # Step 6: individual screening + validation
├── scripts/
│   ├── analyze_quadrant.py      # Step 4: quadrant analysis
│   ├── create_mmlu_eval_set.py  # Step 1: create MMLU eval sets
│   ├── split_eval_set.py        # Step 2: discovery/validation split
│   ├── download_saes.py         # Pre-download SAE weights
│   └── plot/                    # Plotting scripts
│       ├── depth_gradient.py
│       ├── feature_selection.py
│       └── suppression.py
├── src/
│   ├── model.py                 # Model loading, config, layer access
│   ├── evaluation.py            # MCQ evaluation, entropy, flip counting
│   ├── sae.py                   # SAE loading, ResidualStreamHook, SAESuppressionHook
│   ├── quadrant.py              # Quadrant feature loading/classification
│   └── utils.py                 # Seeding, eval set loading, layer parsing
├── results/                     # Generated outputs (gitignored: *.pkl, *.pt)
├── logs/                        # Experiment logs
├── pyproject.toml
└── requirements.txt
```

## How Suppression Works

The `SAESuppressionHook` performs a surgical intervention on the residual stream:

1. At the target layer, intercept the hidden state (last token only)
2. Encode through SAE → sparse feature activations
3. Zero out target feature activations (or scale by `--scale`)
4. Decode back through SAE
5. Compute delta = modified_reconstruction - original_reconstruction
6. Add delta to original hidden state

This preserves all information not captured by the SAE, only removing the specific feature's contribution.

## Notes

- **GPU memory:** Llama-3.1-8B in float16 needs ~16GB. SAEs need ~1GB each (loaded/unloaded one at a time during extraction).
- **Intermediate pickles** (`.pkl`) are pipeline artifacts generated between steps. They're gitignored but necessary — regenerate them by running the pipeline.
- All experiments use `--device cuda` by default. Set `CUDA_VISIBLE_DEVICES` to select your GPU.
- Random seed is fixed at 42 for reproducibility.
