"""
SAE Feature Suppression by Category

Suppress features grouped by quadrant classification (pure_uncertainty,
pure_incorrectness, both) and measure causal impact on accuracy and entropy.

Usage:
    python experiments/suppress_by_category.py \
        --model meta-llama/Llama-3.1-8B \
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_1094.json \
        --quadrant-pkl results/sae_quadrant_1094/quadrant_analysis.pkl \
        --output-dir results/sae_suppression_by_category_1094/
"""

import sys
sys.path.append('.')

import csv
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import gc

from src.utils import seed_everything, load_eval_set, parse_layer_list
from src.model import load_model, get_letter_token_ids, get_default_sae_layers
from src.sae import SAESuppressionHook, load_sae
from src.evaluation import run_eval, compute_flips


def main():
    parser = argparse.ArgumentParser(description="SAE Suppression by Feature Category")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--eval-set', type=str,
                        default='data/eval_sets/eval_set_mcq_arc_challenge_500.json')
    parser.add_argument('--quadrant-pkl', type=str,
                        default='results/sae_quadrant/quadrant_analysis.pkl')
    parser.add_argument('--suppress-layers', type=str, default=None,
                        help='Layers to suppress features at (default: from model config, last 4)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Max features per category per layer')
    parser.add_argument('--scale', type=float, default=0.0,
                        help='Suppression scale (0.0=full suppression)')
    parser.add_argument('--output-dir', type=str,
                        default='results/sae_suppression_by_category/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--both-ranking', type=str, default='unc', choices=['unc', 'min'],
                        help='Ranking for "both" features: unc=by uncertainty effect, min=by min(unc,inc)')

    args = parser.parse_args()
    seed_everything(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer, model_config = load_model(
        args.model, device=args.device, load_in_8bit=args.load_in_8bit
    )

    # Determine suppress layers
    if args.suppress_layers:
        suppress_layers = [int(x) for x in args.suppress_layers.split(',')]
    else:
        default_layers = get_default_sae_layers(model_config)
        suppress_layers = default_layers[-4:]  # last 4 layers by default
    print(f"Suppress layers: {suppress_layers}")

    # Load eval set
    samples = load_eval_set(args.eval_set)

    # Load quadrant-classified features
    print("\nLoading quadrant-classified features...")
    from src.quadrant import load_quadrant_features
    features_by_category = load_quadrant_features(
        args.quadrant_pkl, suppress_layers, args.top_n, args.both_ranking
    )

    letter_token_ids = get_letter_token_ids(tokenizer)
    print(f"Letter token IDs: {letter_token_ids}")

    csv_file = output_dir / "suppression_by_category.csv"

    def save_partial(results):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # =========================================================================
    # Baseline
    # =========================================================================
    print(f"\n{'='*60}")
    print("Baseline (no suppression)")
    print(f"{'='*60}")
    baseline_results = run_eval(model, tokenizer, samples, letter_token_ids,
                                args.device, "Baseline")
    baseline_acc = sum(r['correct'] for r in baseline_results) / len(baseline_results)
    baseline_entropy = np.mean([r['entropy'] for r in baseline_results])
    print(f"Baseline accuracy: {baseline_acc*100:.1f}%  mean entropy: {baseline_entropy:.4f}")

    # =========================================================================
    # Category suppression tests
    # =========================================================================
    categories = ['pure_uncertainty', 'pure_incorrectness', 'both']
    all_results = [{
        'condition': 'baseline', 'category': None, 'layer': None,
        'scale': None, 'n_features': 0, 'features': '',
        'accuracy': baseline_acc, 'acc_delta': 0.0,
        'mean_entropy': baseline_entropy,
        'n_flipped_to_correct': 0, 'n_flipped_to_incorrect': 0,
    }]

    # Per-layer, per-category
    for layer_idx in suppress_layers:
        if layer_idx not in features_by_category:
            continue

        print(f"\nLoading SAE for layer {layer_idx}...")
        sae = load_sae(model_config, layer_idx, device=args.device)

        for category in categories:
            feat_indices = features_by_category[layer_idx][category]
            if not feat_indices:
                print(f"\n  L{layer_idx} [{category}]: no features, skipping")
                continue

            print(f"\n{'='*60}")
            print(f"Layer {layer_idx} | {category} | scale={args.scale} | "
                  f"{len(feat_indices)} features: {feat_indices}")
            print(f"{'='*60}")

            hook = SAESuppressionHook(sae, feat_indices, scale=args.scale)
            hook.register(model, model_config, layer_idx)

            suppressed_results = run_eval(
                model, tokenizer, samples, letter_token_ids, args.device,
                f"L{layer_idx} {category}"
            )
            hook.remove()

            acc = sum(r['correct'] for r in suppressed_results) / len(suppressed_results)
            mean_ent = np.mean([r['entropy'] for r in suppressed_results])
            flipped_correct, flipped_incorrect = compute_flips(baseline_results, suppressed_results)
            delta = (acc - baseline_acc) * 100

            print(f"  Accuracy: {acc*100:.1f}% (delta: {delta:+.1f}%)")
            print(f"  Mean entropy: {mean_ent:.4f}")
            print(f"  Flipped to correct: {flipped_correct}, "
                  f"flipped to incorrect: {flipped_incorrect}")

            all_results.append({
                'condition': f'L{layer_idx}_{category}',
                'category': category, 'layer': layer_idx,
                'scale': args.scale, 'n_features': len(feat_indices),
                'features': str(feat_indices),
                'accuracy': acc, 'acc_delta': delta,
                'mean_entropy': mean_ent,
                'n_flipped_to_correct': flipped_correct,
                'n_flipped_to_incorrect': flipped_incorrect,
            })
            save_partial(all_results)

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    # All layers simultaneously
    print(f"\n{'='*60}")
    print("All layers simultaneously, by category")
    print(f"{'='*60}")

    saes = {}
    for layer_idx in suppress_layers:
        if layer_idx not in features_by_category:
            continue
        saes[layer_idx] = load_sae(model_config, layer_idx, device=args.device)

    for category in categories:
        any_features = any(
            features_by_category.get(l, {}).get(category, [])
            for l in suppress_layers
        )
        if not any_features:
            print(f"\n  All layers [{category}]: no features, skipping")
            continue

        hooks = []
        total_feats = 0
        all_feat_ids = {}
        for layer_idx in suppress_layers:
            if layer_idx not in features_by_category or layer_idx not in saes:
                continue
            feat_indices = features_by_category[layer_idx][category]
            if not feat_indices:
                continue
            hook = SAESuppressionHook(saes[layer_idx], feat_indices, scale=args.scale)
            hook.register(model, model_config, layer_idx)
            hooks.append(hook)
            total_feats += len(feat_indices)
            all_feat_ids[layer_idx] = feat_indices

        print(f"\n{'='*60}")
        print(f"All layers | {category} | scale={args.scale} | {total_feats} total features")
        for l, f in all_feat_ids.items():
            print(f"  L{l}: {f}")
        print(f"{'='*60}")

        suppressed_results = run_eval(
            model, tokenizer, samples, letter_token_ids, args.device,
            f"All {category}"
        )

        for h in hooks:
            h.remove()

        acc = sum(r['correct'] for r in suppressed_results) / len(suppressed_results)
        mean_ent = np.mean([r['entropy'] for r in suppressed_results])
        flipped_correct, flipped_incorrect = compute_flips(baseline_results, suppressed_results)
        delta = (acc - baseline_acc) * 100

        print(f"  Accuracy: {acc*100:.1f}% (delta: {delta:+.1f}%)")
        print(f"  Mean entropy: {mean_ent:.4f}")
        print(f"  Flipped to correct: {flipped_correct}, "
              f"flipped to incorrect: {flipped_incorrect}")

        all_results.append({
            'condition': f'all_layers_{category}',
            'category': category, 'layer': 'all',
            'scale': args.scale, 'n_features': total_feats,
            'features': str(all_feat_ids),
            'accuracy': acc, 'acc_delta': delta,
            'mean_entropy': mean_ent,
            'n_flipped_to_correct': flipped_correct,
            'n_flipped_to_incorrect': flipped_incorrect,
        })
        save_partial(all_results)

    for sae in saes.values():
        del sae
    del saes
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<35} {'Cat':<18} {'#F':>3} {'Acc':>6} {'Delta':>7} "
          f"{'->Corr':>7} {'->Wrong':>8}")
    print("-" * 70)
    for r in all_results:
        cat = r['category'] or ''
        print(f"{r['condition']:<35} {cat:<18} {r['n_features']:>3} "
              f"{r['accuracy']*100:>5.1f}% {r['acc_delta']:>+6.1f}% "
              f"{r['n_flipped_to_correct']:>7} {r['n_flipped_to_incorrect']:>8}")

    save_partial(all_results)
    print(f"\nSaved: {csv_file}")

    pkl_file = output_dir / "suppression_by_category.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': {
                'model': args.model,
                'suppress_layers': suppress_layers,
                'top_n': args.top_n,
                'scale': args.scale,
                'features_by_category': features_by_category,
                'timestamp': datetime.now().isoformat(),
            },
            'results': all_results,
            'baseline_results': baseline_results,
        }, f)
    print(f"Saved: {pkl_file}")


if __name__ == '__main__':
    main()
