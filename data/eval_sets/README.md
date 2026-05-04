# Evaluation Sets

MCQ evaluation sets used in the paper. All files are JSON with a uniform schema:

```json
{
  "metadata": {
    "dataset": "...",
    "format": "mcq",
    "split": "...",
    "num_samples": ...,
    "num_options": 4,
    "seed": 42
  },
  "samples": [
    {
      "id": "...",
      "question": "...",
      "mcq_prompt": "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_letter": "A",
      "correct_answer": "...",
      "distractors": [...]
    }
  ]
}
```

## Files

| File | Dataset | Split | Samples | Role |
|---|---|---|---|---|
| `eval_set_mcq_mmlu_test_14042.json` | MMLU | test (full) | 14,042 | Full MMLU test set |
| `eval_set_mcq_mmlu_test_14042_discovery.json` | MMLU | test (50% split) | 7,021 | Used for feature discovery + screening |
| `eval_set_mcq_mmlu_test_14042_validation.json` | MMLU | test (50% split) | 7,021 | Held-out, used for combined-suppression results |
| `eval_set_mcq_race_train_5000_discovery.json` | RACE (all) | train (subsample) | 5,000 | Used for feature discovery on RACE |
| `eval_set_mcq_race_all_validation_4887.json` | RACE (all) | validation | 4,887 | Held-out, used for combined-suppression results |
| `eval_set_mcq_arc_challenge_2577.json` | ARC-Challenge | all splits | 2,577 | Used for cross-dataset transfer (4-option only) |

## Discovery vs validation

The MMLU test set is split into two halves of 7,021 samples each:
- **Discovery** half is used for the quadrant analysis and individual feature screening
- **Validation** half is used to evaluate the combined suppression of beneficial features

This mirrors a train/test split — features identified on discovery are evaluated for generalization on validation.

For RACE, "discovery" is a 5K subsample of the train split (HF `ehovy/race`, subset `all`, split `train`, seed 42). Validation is the canonical 4,887-sample validation split.

ARC-Challenge is used as a single eval set (no discovery/validation split) for cross-dataset transfer experiments.

## Regenerating

These files can be regenerated from HuggingFace using the scripts in `scripts/`:

```bash
# MMLU
python scripts/create_mmlu_eval_set.py --samples 14042 --split test
python scripts/split_eval_set.py --input data/eval_sets/eval_set_mcq_mmlu_test_14042.json

# RACE
python scripts/create_race_eval_set.py --split train --samples 5000 --seed 42
python scripts/create_race_eval_set.py --split validation
```

ARC was created from `allenai/ai2_arc` `ARC-Challenge`, filtered to 4-option questions only.
