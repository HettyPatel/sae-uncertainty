"""Plot selective prediction accuracy-coverage tradeoff."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='figures/')
    parser.add_argument('--prefix', type=str, default='selective_prediction')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Group by method
    methods = defaultdict(list)
    for r in rows:
        methods[r['method']].append(r)

    # --- Figure 1: Accuracy vs Coverage ---
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'entropy_baseline': '#e74c3c'}
    linestyles = {'entropy_baseline': '--'}
    markers = {'entropy_baseline': 's'}

    sae_methods = [m for m in methods if m.startswith('sae_')]
    sae_colors = ['#e67e22', '#9b59b6', '#3498db', '#2ecc71', '#1abc9c']

    for i, m in enumerate(sae_methods):
        colors[m] = sae_colors[i % len(sae_colors)]
        linestyles[m] = '-'
        markers[m] = 'o'

    # Get baseline accuracy
    baseline_acc = None
    for r in rows:
        if float(r['coverage']) > 0.99:
            baseline_acc = float(r['accuracy']) * 100
            break

    for method, data in methods.items():
        data.sort(key=lambda r: float(r['coverage']))
        coverages = [float(r['coverage']) * 100 for r in data]
        accuracies = [float(r['accuracy']) * 100 for r in data]

        label = method.replace('sae_', 'SAE ').replace('entropy_baseline', 'Entropy baseline')
        ax.plot(coverages, accuracies, f'{markers.get(method, "o")}{linestyles.get(method, "-")}',
                color=colors.get(method, '#333'), linewidth=2, markersize=6,
                alpha=0.85, label=label)

    if baseline_acc:
        ax.axhline(y=baseline_acc, color='grey', linestyle=':', linewidth=1,
                   alpha=0.5, label=f'No abstention ({baseline_acc:.1f}%)')

    ax.set_xlabel('Coverage (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Selective Prediction: Accuracy vs Coverage', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(25, 100)

    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_accuracy_coverage.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_accuracy_coverage.png'}")
    plt.close()

    # --- Figure 2: Breakdown of answered/abstained ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pick the primary SAE method (first one)
    primary_sae = sae_methods[0] if sae_methods else None
    if primary_sae:
        data = sorted(methods[primary_sae], key=lambda r: float(r['threshold']))
        thresholds = [float(r['threshold']) for r in data]
        correct_ans = [int(r['correct_answered']) for r in data]
        incorrect_ans = [int(r['incorrect_answered']) for r in data]
        correct_abs = [int(r['correct_abstained']) for r in data]
        incorrect_abs = [int(r['incorrect_abstained']) for r in data]

        x = np.arange(len(thresholds))
        width = 0.35

        ax1 = axes[0]
        ax1.bar(x - width/2, correct_ans, width, color='#2ecc71', alpha=0.85, label='Correct (answered)')
        ax1.bar(x + width/2, incorrect_ans, width, color='#e74c3c', alpha=0.85, label='Incorrect (answered)')
        ax1.set_xlabel('Confidence Threshold', fontsize=11)
        ax1.set_ylabel('Number of Questions', fontsize=11)
        ax1.set_title('Answered Questions', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[1]
        ax2.bar(x - width/2, correct_abs, width, color='#f39c12', alpha=0.85, label='Correct (abstained)')
        ax2.bar(x + width/2, incorrect_abs, width, color='#3498db', alpha=0.85, label='Incorrect (abstained)')
        ax2.set_xlabel('Confidence Threshold', fontsize=11)
        ax2.set_ylabel('Number of Questions', fontsize=11)
        ax2.set_title('Abstained Questions', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, fontsize=9)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        label = primary_sae.replace('sae_', 'SAE ')
        plt.suptitle(f'Selective Prediction Breakdown ({label})', fontsize=14, y=1.02)

    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_breakdown.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_breakdown.png'}")
    plt.close()


if __name__ == '__main__':
    main()
