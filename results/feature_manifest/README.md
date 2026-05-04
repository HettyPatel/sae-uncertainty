# Feature Manifest

SAE features identified by our quadrant analysis, organized by stage of the pipeline.

## Structure

- **`tested/`** — every feature we ran individual suppression on (with screening flag)
- **`beneficial/`** — features that passed screening (used in combined suppression experiments)

## Pipeline

```
Quadrant analysis (Mann-Whitney U + p<0.05) → tested/ (top-N per layer cap)
                                            ↓
                         Individual suppression on discovery set
                                            ↓
                  Screen: acc_delta ≥ 0 AND ent_delta < 0 → beneficial/
                                            ↓
                 Combined suppression on validation set → paper results
```

## Cap on tested features

For all categories except Llama MMLU "confounded", we tested the **top-5 features per layer** (sorted by effect size). For Llama MMLU "confounded" specifically, we tested all features (no cap, 102 total).

## Counts

### `tested/`
| File | Tested | Beneficial |
|---|---|---|
| `llama_mmlu_confounded.csv` | 102 | 55 |
| `llama_mmlu_pure_uncertainty.csv` | 160 | 63 |
| `llama_mmlu_pure_incorrectness.csv` | 137 | 52 |
| `llama_race_confounded.csv` | 65 | 30 |
| `gemma_mmlu_confounded.csv` | 198 | 39 |
| `gemma_mmlu_pure_uncertainty.csv` | 210 | 45 |
| `gemma_mmlu_pure_incorrectness.csv` | 195 | 70 |
| `gemma_race_confounded.csv` | 130 | 53 |

### `beneficial/`
Same as the "Beneficial" column above — these are the screened features used in the paper's combined suppression results.

## CSV format

### `tested/*.csv`
```
layer,feature_idx,effect_size,acc_delta_pct,ent_delta,beneficial
16,21747,0.5790,+0.0285,-0.005430,True
16,6986,0.3149,+0.0142,-0.000234,True
16,4397,0.0861,+0.0570,+0.000042,False
...
```

- `effect_size` — Cohen's d from quadrant analysis
- `acc_delta_pct` — individual suppression accuracy delta (percentage points) on discovery set
- `ent_delta` — individual suppression mean entropy delta on discovery set
- `beneficial` — True if `acc_delta_pct >= 0` AND `ent_delta < 0`

### `beneficial/*.csv`
```
layer,feature_idx,acc_delta_pct,ent_delta
16,21747,0.0285,-0.005430
16,6986,0.0142,-0.000234
...
```

Subset of `tested/` filtered to `beneficial == True`.

## How to use

Load the CSV, group features by layer, register `SAESuppressionHook` with each group:

```python
import csv
from collections import defaultdict
from src.model import load_model
from src.sae import SAESuppressionHook, load_sae

features_by_layer = defaultdict(list)
with open('results/feature_manifest/beneficial/llama_mmlu_confounded.csv') as f:
    for row in csv.DictReader(f):
        features_by_layer[int(row['layer'])].append(int(row['feature_idx']))

model, tokenizer, config = load_model('meta-llama/Llama-3.1-8B')

hooks = []
for layer_idx, feats in features_by_layer.items():
    sae = load_sae(config, layer_idx)
    hook = SAESuppressionHook(sae, feats, scale=0.0)
    hook.register(model, config, layer_idx)
    hooks.append(hook)

# Now run inference — the hooks will suppress these features
# ...
for h in hooks:
    h.remove()
```
