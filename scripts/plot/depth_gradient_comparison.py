"""Plot depth gradient comparison between two models (e.g. Llama vs Gemma)."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def get_counts(data, p_threshold):
    layers = sorted(data['all_comparisons'].keys())
    n_unc, n_inc, n_both = [], [], []
    top_unc, top_inc, top_both = [], [], []

    for l in layers:
        comp = data['all_comparisons'][l]
        sig_unc = {r['feature_idx'] for r in comp['pure_uncertainty']
                   if r['p_value'] < p_threshold and r['effect_size'] > 0}
        sig_inc = {r['feature_idx'] for r in comp['pure_incorrectness']
                   if r['p_value'] < p_threshold and r['effect_size'] > 0}

        both_set = sig_unc & sig_inc
        n_unc.append(len(sig_unc - sig_inc))
        n_inc.append(len(sig_inc - sig_unc))
        n_both.append(len(both_set))

        unc_sig = [r for r in comp['pure_uncertainty']
                   if r['p_value'] < p_threshold and r['effect_size'] > 0]
        inc_sig = [r for r in comp['pure_incorrectness']
                   if r['p_value'] < p_threshold and r['effect_size'] > 0]

        # For "both", use the min(unc_effect, inc_effect) as the feature's strength
        inc_lookup = {r['feature_idx']: r['effect_size'] for r in inc_sig}
        both_effects = [min(r['effect_size'], inc_lookup.get(r['feature_idx'], 0))
                        for r in unc_sig if r['feature_idx'] in both_set]

        top_unc.append(max((r['effect_size'] for r in unc_sig), default=0))
        top_inc.append(max((r['effect_size'] for r in inc_sig), default=0))
        top_both.append(max(both_effects, default=0))

    return layers, n_unc, n_inc, n_both, top_unc, top_inc, top_both


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl1', type=str, required=True, help='First model quadrant pkl')
    parser.add_argument('--pkl2', type=str, required=True, help='Second model quadrant pkl')
    parser.add_argument('--name1', type=str, default='Llama-3.1-8B')
    parser.add_argument('--name2', type=str, default='Gemma-2-9B')
    parser.add_argument('--output-dir', type=str, default='figures/')
    parser.add_argument('--prefix', type=str, default='depth_comparison')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d1 = pickle.load(open(args.pkl1, 'rb'))
    d2 = pickle.load(open(args.pkl2, 'rb'))

    p = d1['config']['p_threshold']
    l1, unc1, inc1, both1, tunc1, tinc1, tboth1 = get_counts(d1, p)
    l2, unc2, inc2, both2, tunc2, tinc2, tboth2 = get_counts(d2, p)

    # Normalize layer positions to [0, 1] for comparison
    norm1 = np.array(l1) / max(l1)
    norm2 = np.array(l2) / max(l2)

    # --- Figure 1: Feature counts by category (normalized depth) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, cat, v1, v2, color in [
        (axes[0], 'Pure Uncertainty', unc1, unc2, '#e74c3c'),
        (axes[1], 'Pure Incorrectness', inc1, inc2, '#3498db'),
        (axes[2], 'Both', both1, both2, '#9b59b6'),
    ]:
        ax.plot(norm1, v1, 'o-', color=color, alpha=0.7, markersize=4,
                linewidth=1.5, label=args.name1)
        ax.plot(norm2, v2, 's--', color=color, alpha=0.7, markersize=4,
                linewidth=1.5, label=args.name2)
        ax.set_title(cat, fontsize=13)
        ax.set_xlabel('Normalized Depth (layer / max_layer)', fontsize=11)
        ax.set_ylabel('# Significant Features', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Feature Classification by Depth: Llama vs Gemma (p25 threshold)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_counts.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_counts.png'}")
    plt.close()

    # --- Figure 2: Peak effect sizes (normalized depth) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.plot(norm1, tunc1, 'o-', color='#e74c3c', alpha=0.7, markersize=4,
             linewidth=1.5, label=args.name1)
    ax1.plot(norm2, tunc2, 's--', color='#e74c3c', alpha=0.7, markersize=4,
             linewidth=1.5, label=args.name2)
    ax1.set_title('Pure Uncertainty (C vs A)', fontsize=13)
    ax1.set_xlabel('Normalized Depth', fontsize=11)
    ax1.set_ylabel("Peak Cohen's d", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(norm1, tinc1, 'o-', color='#3498db', alpha=0.7, markersize=4,
             linewidth=1.5, label=args.name1)
    ax2.plot(norm2, tinc2, 's--', color='#3498db', alpha=0.7, markersize=4,
             linewidth=1.5, label=args.name2)
    ax2.set_title('Pure Incorrectness (B vs A)', fontsize=13)
    ax2.set_xlabel('Normalized Depth', fontsize=11)
    ax2.set_ylabel("Peak Cohen's d", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3.plot(norm1, tboth1, 'o-', color='#9b59b6', alpha=0.7, markersize=4,
             linewidth=1.5, label=args.name1)
    ax3.plot(norm2, tboth2, 's--', color='#9b59b6', alpha=0.7, markersize=4,
             linewidth=1.5, label=args.name2)
    ax3.set_title('Both (min of C vs A, B vs A)', fontsize=13)
    ax3.set_xlabel('Normalized Depth', fontsize=11)
    ax3.set_ylabel("Peak Cohen's d", fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Peak Effect Size by Depth: Llama vs Gemma (p25 threshold)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_effect_sizes.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_effect_sizes.png'}")
    plt.close()

    # --- Figure 3: "Both" features proportion (normalized) ---
    fig, ax = plt.subplots(figsize=(10, 5))

    total1 = [u + i + b for u, i, b in zip(unc1, inc1, both1)]
    total2 = [u + i + b for u, i, b in zip(unc2, inc2, both2)]
    prop1 = [b / t * 100 if t > 0 else 0 for b, t in zip(both1, total1)]
    prop2 = [b / t * 100 if t > 0 else 0 for b, t in zip(both2, total2)]

    ax.plot(norm1, prop1, 'o-', color='#9b59b6', alpha=0.7, markersize=4,
            linewidth=1.5, label=args.name1)
    ax.plot(norm2, prop2, 's--', color='#9b59b6', alpha=0.7, markersize=4,
            linewidth=1.5, label=args.name2)
    ax.set_title('"Both" Features as % of All Significant Features', fontsize=13)
    ax.set_xlabel('Normalized Depth (layer / max_layer)', fontsize=11)
    ax.set_ylabel('% Both', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_both_proportion.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_both_proportion.png'}")
    plt.close()


if __name__ == '__main__':
    main()
