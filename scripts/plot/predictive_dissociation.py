"""Plot per-layer correctness prediction AUROC by feature category."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to prediction_results.csv')
    parser.add_argument('--output-dir', type=str, default='figures/')
    parser.add_argument('--prefix', type=str, default='predictive_dissociation')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Get entropy baseline
    bl = next(r for r in rows if r['feature_set'] == 'entropy_baseline')
    baseline_auroc = float(bl['metric1'])

    # Collect per-layer correctness AUROC
    feature_sets = ['pure_uncertainty', 'pure_incorrectness', 'both']
    colors = {'pure_uncertainty': '#9b59b6', 'pure_incorrectness': '#95a5a6', 'both': '#e67e22'}
    labels = {'pure_uncertainty': 'Pure uncertainty', 'pure_incorrectness': 'Pure incorrectness', 'both': 'Confounded (both)'}
    markers = {'pure_uncertainty': 'o', 'pure_incorrectness': 's', 'both': 'D'}

    data = defaultdict(dict)  # {fs: {layer: auroc}}
    for r in rows:
        if r['target'] == 'correctness' and r['scope'].startswith('L') and r['feature_set'] in feature_sets:
            layer = int(r['layer'])
            data[r['feature_set']][layer] = float(r['metric1'])

    # --- Figure: Correctness AUROC by layer ---
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    for fs in feature_sets:
        layers_sorted = sorted(data[fs].keys())
        aurocs = [data[fs][l] for l in layers_sorted]
        # Replace 0.0 (missing) with NaN so lines break
        aurocs_plot = [a if a > 0.01 else np.nan for a in aurocs]
        ax.plot(layers_sorted, aurocs_plot, f'{markers[fs]}-', color=colors[fs],
                linewidth=1.5, markersize=3.5, alpha=0.85, label=labels[fs])

    ax.axhline(y=baseline_auroc, color='#e74c3c', linestyle='--', linewidth=1.5,
               alpha=0.8, label=f'Entropy baseline ({baseline_auroc:.3f})')

    ax.set_xlabel('Layer', fontsize=9)
    ax.set_ylabel('Correctness Prediction AUROC', fontsize=9)
    ax.set_title('Predictive Dissociation:\nPer-Layer Correctness AUROC', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(0.45, 0.85)
    ax.set_xticks(range(0, 32, 4))
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_correctness.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_correctness.png'}")
    plt.close()

    # --- Figure: Entropy Spearman by layer ---
    data_ent = defaultdict(dict)
    for r in rows:
        if r['target'] == 'entropy' and r['scope'].startswith('L') and r['feature_set'] in feature_sets:
            layer = int(r['layer'])
            data_ent[r['feature_set']][layer] = float(r['metric1'])

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    for fs in feature_sets:
        layers_sorted = sorted(data_ent[fs].keys())
        spearman = [data_ent[fs][l] for l in layers_sorted]
        spearman_plot = [s if abs(s) > 0.01 else np.nan for s in spearman]
        ax.plot(layers_sorted, spearman_plot, f'{markers[fs]}-', color=colors[fs],
                linewidth=1.5, markersize=3.5, alpha=0.85, label=labels[fs])

    ax.set_xlabel('Layer', fontsize=9)
    ax.set_ylabel('Entropy Prediction (Spearman ρ)', fontsize=9)
    ax.set_title('Predictive Dissociation:\nPer-Layer Entropy Prediction', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 31.5)
    ax.set_xticks(range(0, 32, 4))
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_entropy.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_entropy.png'}")
    plt.close()


if __name__ == '__main__':
    main()
