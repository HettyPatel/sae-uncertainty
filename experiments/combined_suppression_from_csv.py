"""
Combined suppression using beneficial features from individual suppression CSV.

Reads an individual_suppression_by_category.csv, selects beneficial features
(acc >= baseline AND ent < baseline), and runs combined suppression per-layer
and all-layers on a given eval set.

Usage:
    python experiments/combined_suppression_from_csv.py \
        --model google/gemma-2-9b \
        --individual-csv results/sae_individual_suppression_both_gemma_p25/individual_suppression_by_category.csv \
        --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
        --output-dir results/sae_individual_suppression_both_gemma_p25/
"""

import csv
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse
import gc

from src.utils import seed_everything, load_eval_set
from src.model import load_model, get_letter_token_ids
from src.sae import SAESuppressionHook, load_sae
from src.evaluation import run_eval, summarise, compute_flips


def main():
    parser = argparse.ArgumentParser(description="Combined Suppression from CSV")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--individual-csv', type=str, required=True)
    parser.add_argument('--eval-set', type=str, required=True)
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load beneficial features from CSV
    selected = defaultdict(list)
    with open(args.individual_csv) as f:
        for row in csv.DictReader(f):
            if float(row['acc_delta']) >= 0 and float(row['ent_delta']) < 0:
                selected[int(row['layer'])].append(int(row['feature_idx']))
    selected = dict(selected)

    total = sum(len(v) for v in selected.values())
    print(f"Beneficial features: {total} across {len(selected)} layers")
    for l, feats in sorted(selected.items()):
        print(f"  L{l}: {feats}")

    if total == 0:
        print("No beneficial features. Exiting.")
        return

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, model_config = load_model(args.model, device=args.device)
    letter_token_ids = get_letter_token_ids(tokenizer)

    # Load eval set
    samples = load_eval_set(args.eval_set)

    # Baseline
    print(f"\nBaseline (no suppression)")
    baseline = run_eval(model, tokenizer, samples, letter_token_ids,
                        args.device, "Baseline")
    base_acc, base_ent = summarise(baseline)
    print(f"Baseline accuracy: {base_acc*100:.2f}%  mean entropy: {base_ent:.4f}")

    all_rows = []
    csv_file = output_dir / "combined_suppression.csv"

    def save_partial():
        if all_rows:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)

    # Per-layer suppression
    for layer_idx, feats in sorted(selected.items()):
        sae = load_sae(model_config, layer_idx, device=args.device)
        h = SAESuppressionHook(sae, feats, scale=args.scale)
        h.register(model, model_config, layer_idx)

        res = run_eval(model, tokenizer, samples, letter_token_ids,
                       args.device, f"L{layer_idx}")
        h.remove()
        del sae
        torch.cuda.empty_cache()
        gc.collect()

        acc, ent = summarise(res)
        flipped_correct, flipped_incorrect = compute_flips(baseline, res)
        acc_delta = (acc - base_acc) * 100
        ent_delta = ent - base_ent

        print(f"  L{layer_idx} ({len(feats)} feats): "
              f"acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
              f"ent={ent:.4f} ({ent_delta:+.4f})  "
              f"->correct={flipped_correct} ->wrong={flipped_incorrect}")

        all_rows.append({
            'condition': f'L{layer_idx}_combined',
            'layer': layer_idx,
            'features': str(feats),
            'n_features': len(feats),
            'accuracy': acc, 'acc_delta': acc_delta,
            'mean_entropy': ent, 'ent_delta': ent_delta,
            'n_flipped_to_correct': flipped_correct,
            'n_flipped_to_incorrect': flipped_incorrect,
        })
        save_partial()

    # All layers combined
    hooks = []
    saes = {}
    for layer_idx, feats in selected.items():
        sae = load_sae(model_config, layer_idx, device=args.device)
        saes[layer_idx] = sae
        h = SAESuppressionHook(sae, feats, scale=args.scale)
        h.register(model, model_config, layer_idx)
        hooks.append(h)

    res = run_eval(model, tokenizer, samples, letter_token_ids,
                   args.device, "All layers")
    for h in hooks:
        h.remove()

    for sae in saes.values():
        del sae
    del saes
    torch.cuda.empty_cache()
    gc.collect()

    acc, ent = summarise(res)
    flipped_correct, flipped_incorrect = compute_flips(baseline, res)
    acc_delta = (acc - base_acc) * 100
    ent_delta = ent - base_ent

    print(f"\n  ALL LAYERS ({total} feats): "
          f"acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
          f"ent={ent:.4f} ({ent_delta:+.4f})  "
          f"->correct={flipped_correct} ->wrong={flipped_incorrect}")

    all_rows.append({
        'condition': 'all_layers_combined',
        'layer': 'all',
        'features': str(selected),
        'n_features': total,
        'accuracy': acc, 'acc_delta': acc_delta,
        'mean_entropy': ent, 'ent_delta': ent_delta,
        'n_flipped_to_correct': flipped_correct,
        'n_flipped_to_incorrect': flipped_incorrect,
    })
    save_partial()

    print(f"\nSaved: {csv_file}")
    print(f"\nDONE")


if __name__ == '__main__':
    main()
