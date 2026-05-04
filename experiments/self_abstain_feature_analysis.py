"""
Self-Abstention Feature Activation Analysis

Compare SAE feature activations between 4-option and 5-option (with abstain option) prompts.
Check if adding "I don't know" changes activation of uncertainty/incorrectness/both features.

Usage:
    python experiments/self_abstain_feature_analysis.py \
        --model meta-llama/Llama-3.1-8B \
        --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
        --quadrant-pkl results/sae_quadrant_mmlu_discovery_all_layers_p25/quadrant_analysis.pkl \
        --self-abstain-results results/sae_self_abstain_experiment/self_abstain_results.csv \
        --layers 16,18,20,24 \
        --output-dir results/sae_self_abstain_feature_analysis/
"""

import csv
import json
import torch
import pickle
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy import stats

from src.utils import seed_everything, load_eval_set
from src.model import load_model, get_letter_token_ids
from src.sae import load_sae


def get_feature_categories(quadrant_pkl, layers):
    with open(quadrant_pkl, 'rb') as f:
        data = pickle.load(f)
    p = data['config']['p_threshold']
    categories = {}
    for layer in layers:
        if layer not in data['all_comparisons']:
            continue
        comp = data['all_comparisons'][layer]
        sig_unc = {r['feature_idx'] for r in comp['pure_uncertainty']
                   if r['p_value'] < p and r['effect_size'] > 0}
        sig_inc = {r['feature_idx'] for r in comp['pure_incorrectness']
                   if r['p_value'] < p and r['effect_size'] > 0}
        categories[layer] = {
            'pure_uncertainty': sorted(sig_unc - sig_inc),
            'pure_incorrectness': sorted(sig_inc - sig_unc),
            'both': sorted(sig_unc & sig_inc),
        }
    return categories


def add_abstain_option(prompt):
    return prompt.replace("\nAnswer:", "\nE. I don't know\nAnswer:")


def main():
    parser = argparse.ArgumentParser(description="Self-Abstention Feature Activation Analysis")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--eval-set', type=str, required=True)
    parser.add_argument('--quadrant-pkl', type=str, required=True)
    parser.add_argument('--self-abstain-results', type=str, required=True,
                        help='CSV from self_abstain_experiment.py with chose_abstain column')
    parser.add_argument('--layers', type=str, default='16,18,20,24')
    parser.add_argument('--output-dir', type=str, default='results/sae_self_abstain_feature_analysis/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = [int(x) for x in args.layers.split(',')]

    # Load self-abstain results to know which questions chose to abstain
    abstain_data = {}
    with open(args.self_abstain_results) as f:
        for row in csv.DictReader(f):
            abstain_data[row['id']] = {
                'chose_abstain': row['chose_abstain'] == 'True',
                'correct_4opt': row['correct_4opt'] == 'True',
            }

    # Load feature categories
    categories = get_feature_categories(args.quadrant_pkl, layers)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, model_config = load_model(args.model, device=args.device)

    # Load eval set
    samples = load_eval_set(args.eval_set)

    # For each layer, load SAE and collect activations for both prompt types
    all_results = []

    for layer_idx in layers:
        print(f"\nLayer {layer_idx}")
        sae = load_sae(model_config, layer_idx, device=args.device)
        cat = categories[layer_idx]

        # Collect all significant feature indices for this layer
        all_feat_indices = set()
        for cat_name in ['pure_uncertainty', 'pure_incorrectness', 'both']:
            all_feat_indices.update(cat[cat_name])
        all_feat_indices = sorted(all_feat_indices)
        feat_to_col = {f: i for i, f in enumerate(all_feat_indices)}

        # Collect activations
        acts_4opt = np.zeros((len(samples), len(all_feat_indices)))
        acts_5opt = np.zeros((len(samples), len(all_feat_indices)))

        for q_idx, sample in enumerate(tqdm(samples, desc=f"L{layer_idx}", leave=False)):
            prompt_4 = sample.get('mcq_prompt', sample.get('prompt', ''))
            prompt_5 = add_abstain_option(prompt_4)

            for prompt, acts_array in [(prompt_4, acts_4opt), (prompt_5, acts_5opt)]:
                inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
                hook_data = {}

                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hook_data['hidden'] = output[0][:, -1, :].detach()
                    else:
                        hook_data['hidden'] = output[:, -1, :].detach()

                from src.model import get_layers
                model_layers = get_layers(model, model_config)
                handle = model_layers[layer_idx].register_forward_hook(hook_fn)

                with torch.no_grad():
                    model(**inputs)

                handle.remove()

                hidden = hook_data['hidden'].float()
                with torch.no_grad():
                    feat_acts = sae.encode(hidden).squeeze(0).cpu().numpy()

                for feat_idx in all_feat_indices:
                    acts_array[q_idx, feat_to_col[feat_idx]] = feat_acts[feat_idx]

        del sae
        torch.cuda.empty_cache()

        # Analysis per category
        chose_abstain = np.array([abstain_data[s['id']]['chose_abstain'] for s in samples])
        correct_4 = np.array([abstain_data[s['id']]['correct_4opt'] for s in samples])

        for cat_name in ['pure_uncertainty', 'pure_incorrectness', 'both']:
            feat_indices = cat[cat_name]
            if not feat_indices:
                continue
            cols = [feat_to_col[f] for f in feat_indices]

            # Mean activation across category features
            mean_4opt = acts_4opt[:, cols].mean(axis=1)
            mean_5opt = acts_5opt[:, cols].mean(axis=1)
            delta = mean_5opt - mean_4opt

            # Split by abstention choice and correctness
            groups = {
                'all': np.ones(len(samples), dtype=bool),
                'chose_abstain': chose_abstain,
                'not_abstain': ~chose_abstain,
                'correct_4opt': correct_4,
                'incorrect_4opt': ~correct_4,
                'abstain_and_correct': chose_abstain & correct_4,
                'abstain_and_incorrect': chose_abstain & ~correct_4,
            }

            print(f"\n  {cat_name} ({len(feat_indices)} features):")
            for group_name, mask in groups.items():
                n = mask.sum()
                if n == 0:
                    continue
                m4 = mean_4opt[mask].mean()
                m5 = mean_5opt[mask].mean()
                d = delta[mask].mean()
                # Test if delta is significantly different from 0
                if n > 1:
                    t_stat, p_val = stats.ttest_1samp(delta[mask], 0)
                else:
                    p_val = 1.0
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"    {group_name:<22}: n={n:>5}  4opt={m4:.4f}  5opt={m5:.4f}  delta={d:+.4f}  p={p_val:.4f} {sig}")

                all_results.append({
                    'layer': layer_idx,
                    'category': cat_name,
                    'n_features': len(feat_indices),
                    'group': group_name,
                    'n_samples': int(n),
                    'mean_act_4opt': m4,
                    'mean_act_5opt': m5,
                    'mean_delta': d,
                    'p_value': p_val,
                })

    # Save
    csv_file = output_dir / "self_abstain_feature_analysis.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {csv_file}")
    print("DONE")


if __name__ == '__main__':
    main()
