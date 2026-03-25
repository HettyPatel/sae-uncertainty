"""Plot per-layer feature selection results: accuracy and entropy with baseline reference."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to validation_suppression.csv')
    parser.add_argument('--output-dir', type=str, default='figures/')
    parser.add_argument('--prefix', type=str, default='feature_selection')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse CSV
    rows = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Separate per-layer and all-layers
    per_layer = [r for r in rows if r['condition'] != 'all_layers_combined']
    combined = [r for r in rows if r['condition'] == 'all_layers_combined'][0]

    layers = [int(r['layer']) for r in per_layer]
    n_features = [int(r['n_features']) for r in per_layer]
    accs = [float(r['accuracy']) * 100 for r in per_layer]
    ents = [float(r['mean_entropy']) for r in per_layer]

    # Baseline = accuracy - delta (recover from first row)
    base_acc = (float(per_layer[0]['accuracy']) - float(per_layer[0]['acc_delta']) / 100) * 100
    base_ent = float(per_layer[0]['mean_entropy']) - float(per_layer[0]['ent_delta'])

    combined_acc = float(combined['accuracy']) * 100
    combined_ent = float(combined['mean_entropy'])

    # Labels
    labels = [f"L{l}\n({n})" for l, n in zip(layers, n_features)]
    labels.append(f"All\n({combined['n_features']})")

    accs.append(combined_acc)
    ents.append(combined_ent)

    x = np.arange(len(labels))

    # --- Figure: Accuracy ---
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#3498db'] * len(per_layer) + ['#e74c3c']
    bars = ax.bar(x, accs, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=base_acc, color='black', linestyle='--', linewidth=1.5, label=f'Baseline ({base_acc:.2f}%)')
    ax.set_xlabel('Layer (# features suppressed)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy After Suppressing Selected "Both" Features', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, accs)):
        delta = val - base_acc
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{delta:+.2f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_accuracy.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_accuracy.png'}")
    plt.close()

    # --- Figure: Entropy ---
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(x, ents, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=base_ent, color='black', linestyle='--', linewidth=1.5, label=f'Baseline ({base_ent:.4f})')
    ax.set_xlabel('Layer (# features suppressed)', fontsize=12)
    ax.set_ylabel('Mean Entropy', fontsize=12)
    ax.set_title('Entropy After Suppressing Selected "Both" Features', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, ents)):
        delta = val - base_ent
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{delta:+.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_entropy.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_entropy.png'}")
    plt.close()


if __name__ == '__main__':
    main()
