"""
Plot Experiment 30 both-feature selection results.

Usage:
    python scripts/plot_both_feature_selection.py \
        --results-dir results/sae_both_feature_selection_no31 \
        --output-dir figures/sae_both_feature_selection_no31
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('n_features', 'n_flipped_to_correct', 'n_flipped_to_incorrect'):
            if k in r and r[k]:
                r[k] = int(float(r[k]))
        for k in ('accuracy', 'acc_delta', 'mean_entropy', 'ent_delta',
                   'unc_effect', 'inc_effect', 'acc_selection', 'acc_delta_selection',
                   'ent_selection', 'ent_delta_selection'):
            if k in r and r[k]:
                r[k] = float(r[k])
    return rows


def plot_individual_selection(individual_rows, output_dir):
    """One figure per layer: bar chart of each feature's acc_delta and ent_delta,
    colored by beneficial/not."""
    layers = sorted(set(int(r['layer']) for r in individual_rows))

    for layer in layers:
        layer_rows = [r for r in individual_rows if int(r['layer']) == layer]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Layer {layer} — Individual Feature Selection (Discovery Set)',
                     fontsize=14, fontweight='bold')

        features = [str(r['feature_idx']) for r in layer_rows]
        acc_deltas = [r['acc_delta_selection'] for r in layer_rows]
        ent_deltas = [r['ent_delta_selection'] for r in layer_rows]
        beneficial = [r['beneficial'] == 'True' if isinstance(r['beneficial'], str)
                      else r['beneficial'] for r in layer_rows]
        colors = ['#4CAF50' if b else '#F44336' for b in beneficial]

        x = np.arange(len(features))

        # Accuracy delta
        bars1 = ax1.bar(x, acc_deltas, color=colors, edgecolor='white', linewidth=0.5)
        ax1.set_xlabel('Feature Index', fontsize=11)
        ax1.set_ylabel('Accuracy Delta (%)', fontsize=11)
        ax1.set_title('Accuracy Change', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
        ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax1.grid(axis='y', alpha=0.3)

        # Entropy delta
        bars2 = ax2.bar(x, ent_deltas, color=colors, edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('Feature Index', fontsize=11)
        ax2.set_ylabel('Entropy Delta', fontsize=11)
        ax2.set_title('Entropy Change', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
        ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax2.grid(axis='y', alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#4CAF50', label='Beneficial (selected)'),
                           Patch(facecolor='#F44336', label='Not beneficial')]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2,
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        fig.savefig(output_dir / f'individual_selection_L{layer}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved individual_selection_L{layer}.png')


def plot_validation_summary(validation_rows, output_dir):
    """Bar chart: per-layer combined + all-layers, showing accuracy delta and entropy delta."""
    # Separate per-layer and all-layers
    per_layer = [r for r in validation_rows if r['layer'] != 'all']
    all_layers = [r for r in validation_rows if r['layer'] == 'all']

    labels = [f"L{r['layer']}" for r in per_layer]
    if all_layers:
        labels.append('All Layers')
        per_layer = per_layer + all_layers

    n = len(labels)
    x = np.arange(n)

    acc_deltas = [r['acc_delta'] for r in per_layer]
    ent_deltas = [r['ent_delta'] for r in per_layer]
    n_features = [r['n_features'] for r in per_layer]
    flipped_correct = [r['n_flipped_to_correct'] for r in per_layer]
    flipped_incorrect = [r['n_flipped_to_incorrect'] for r in per_layer]

    # ── Figure 1: Accuracy and Entropy Deltas ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Validation Set — Combined Beneficial Features by Layer',
                 fontsize=14, fontweight='bold')

    # Color the all-layers bar differently
    colors_acc = ['#2196F3'] * (n - 1) + ['#FF9800'] if all_layers else ['#2196F3'] * n
    colors_ent = ['#9C27B0'] * (n - 1) + ['#FF9800'] if all_layers else ['#9C27B0'] * n

    bars1 = ax1.bar(x, acc_deltas, color=colors_acc, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Accuracy Delta (%)', fontsize=11)
    ax1.set_title('Accuracy Change from Suppression', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.grid(axis='y', alpha=0.3)
    # Annotate with number of features
    for i, (bar, nf) in enumerate(zip(bars1, n_features)):
        ax1.annotate(f'{nf}F', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    bars2 = ax2.bar(x, ent_deltas, color=colors_ent, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Entropy Delta', fontsize=11)
    ax2.set_title('Entropy Change from Suppression', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, nf) in enumerate(zip(bars2, n_features)):
        ax2.annotate(f'{nf}F', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='top' if bar.get_height() < 0 else 'bottom',
                     fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'validation_combined_deltas.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved validation_combined_deltas.png')

    # ── Figure 2: Flipped questions ──
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Validation Set — Questions Flipped by Suppression',
                 fontsize=14, fontweight='bold')

    width = 0.35
    bars_c = ax.bar(x - width / 2, flipped_correct, width, label='Flipped to correct',
                    color='#4CAF50', edgecolor='white')
    bars_w = ax.bar(x + width / 2, flipped_incorrect, width, label='Flipped to incorrect',
                    color='#F44336', edgecolor='white')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Number of Questions', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Annotate net
    for i in range(n):
        net = flipped_correct[i] - flipped_incorrect[i]
        sign = '+' if net >= 0 else ''
        y_pos = max(flipped_correct[i], flipped_incorrect[i]) + 5
        ax.annotate(f'net: {sign}{net}', (x[i], y_pos),
                    ha='center', fontsize=9, fontweight='bold',
                    color='#4CAF50' if net >= 0 else '#F44336')

    plt.tight_layout()
    fig.savefig(output_dir / 'validation_flipped_questions.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved validation_flipped_questions.png')

    # ── Figure 3: Absolute accuracy and entropy ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Validation Set — Absolute Accuracy & Entropy After Suppression',
                 fontsize=14, fontweight='bold')

    accuracies = [r['accuracy'] * 100 for r in per_layer]
    entropies = [r['mean_entropy'] for r in per_layer]

    # Baseline (first row's accuracy - delta gives baseline)
    baseline_acc = (per_layer[0]['accuracy'] - per_layer[0]['acc_delta'] / 100) * 100
    baseline_ent = per_layer[0]['mean_entropy'] - per_layer[0]['ent_delta']

    bars1 = ax1.bar(x, accuracies, color=colors_acc, edgecolor='white', linewidth=0.5)
    ax1.axhline(y=baseline_acc, color='red', linewidth=1.5, linestyle='--',
                label=f'Baseline ({baseline_acc:.1f}%)')
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Accuracy After Suppression', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(min(accuracies) - 2, max(max(accuracies), baseline_acc) + 2)

    bars2 = ax2.bar(x, entropies, color=colors_ent, edgecolor='white', linewidth=0.5)
    ax2.axhline(y=baseline_ent, color='red', linewidth=1.5, linestyle='--',
                label=f'Baseline ({baseline_ent:.3f})')
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Mean Entropy', fontsize=11)
    ax2.set_title('Entropy After Suppression', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'validation_absolute_values.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved validation_absolute_values.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    individual_csv = args.results_dir / 'individual_suppression_selection.csv'
    validation_csv = args.results_dir / 'validation_suppression.csv'

    print('Loading results...')
    individual_rows = load_csv(individual_csv)
    validation_rows = load_csv(validation_csv)

    print('Plotting individual feature selection...')
    plot_individual_selection(individual_rows, args.output_dir)

    print('Plotting validation summary...')
    plot_validation_summary(validation_rows, args.output_dir)

    print('Done!')


if __name__ == '__main__':
    main()
