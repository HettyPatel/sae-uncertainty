"""
Predict Model Incorrectness and Entropy from SAE Feature Activations

Train on discovery set feature activations to predict:
  1. Correctness: binary classification (logistic regression, AUROC)
  2. Entropy: continuous regression (ridge regression, Spearman r, R²)

Compare predictive power of different feature categories
(pure_uncertainty, pure_incorrectness, both) per layer and across all layers.
Includes entropy-only baseline for correctness prediction.

Usage:
    python experiments/predict_incorrectness.py \
        --discovery-pkl results/sae_uncertainty_mmlu_discovery_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
        --validation-pkl results/sae_uncertainty_mmlu_validation_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
        --quadrant-pkl results/sae_quadrant_mmlu_discovery_all_layers_p25/quadrant_analysis.pkl \
        --output-dir results/sae_predict_incorrectness_p25/
"""

import csv
import pickle
import numpy as np
from pathlib import Path
import argparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


def load_feature_activations(pkl_path):
    """Load per-question data from feature extraction pkl."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['per_question_data'], data['config']['sae_layers']


def get_feature_categories(quadrant_pkl, layers):
    """Get feature indices per category per layer from quadrant analysis."""
    with open(quadrant_pkl, 'rb') as f:
        data = pickle.load(f)

    p_threshold = data['config']['p_threshold']
    categories = {}

    for layer in layers:
        if layer not in data['all_comparisons']:
            continue

        comp = data['all_comparisons'][layer]

        sig_unc = {r['feature_idx'] for r in comp['pure_uncertainty']
                   if r['p_value'] < p_threshold and r['effect_size'] > 0}
        sig_inc = {r['feature_idx'] for r in comp['pure_incorrectness']
                   if r['p_value'] < p_threshold and r['effect_size'] > 0}

        categories[layer] = {
            'pure_uncertainty': sorted(sig_unc - sig_inc),
            'pure_incorrectness': sorted(sig_inc - sig_unc),
            'both': sorted(sig_unc & sig_inc),
        }

    return categories


def build_feature_matrix(per_question_data, layer, feature_indices):
    """Build a dense feature matrix for given questions and feature indices."""
    if not feature_indices:
        return np.zeros((len(per_question_data), 0))

    feat_to_col = {f: i for i, f in enumerate(feature_indices)}
    X = np.zeros((len(per_question_data), len(feature_indices)))

    for q_idx, q in enumerate(per_question_data):
        if layer in q['sae_features']:
            sf = q['sae_features'][layer]
            for idx, val in zip(sf['indices'], sf['values']):
                if idx in feat_to_col:
                    X[q_idx, feat_to_col[idx]] = val

    return X


def build_correctness_labels(per_question_data):
    return np.array([1 if q['correct'] else 0 for q in per_question_data])


def build_entropy_labels(per_question_data):
    return np.array([q['entropy'] for q in per_question_data])


def evaluate_classification(X_train, y_train, X_test, y_test):
    """Train logistic regression, return AUROC, accuracy, F1."""
    if X_train.shape[1] == 0:
        return {'auroc': 0.5, 'accuracy': 0.0, 'f1': 0.0, 'n_features': 0}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    clf.fit(X_train_s, y_train)

    y_prob = clf.predict_proba(X_test_s)[:, 1]
    y_pred = clf.predict(X_test_s)

    return {
        'auroc': roc_auc_score(y_test, y_prob),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'n_features': X_train.shape[1],
    }


def evaluate_regression(X_train, y_train, X_test, y_test):
    """Train ridge regression, return Spearman r, R²."""
    if X_train.shape[1] == 0:
        return {'spearman_r': 0.0, 'spearman_p': 1.0, 'r2': 0.0, 'n_features': 0}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(X_train_s, y_train)

    y_pred = reg.predict(X_test_s)

    sp_r, sp_p = spearmanr(y_test, y_pred)

    return {
        'spearman_r': sp_r,
        'spearman_p': sp_p,
        'r2': r2_score(y_test, y_pred),
        'n_features': X_train.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Incorrectness and Entropy from SAE Features")
    parser.add_argument('--discovery-pkl', type=str, required=True)
    parser.add_argument('--validation-pkl', type=str, required=True)
    parser.add_argument('--quadrant-pkl', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results/sae_predict_incorrectness_p25/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading discovery set...")
    train_data, train_layers = load_feature_activations(args.discovery_pkl)
    print(f"  {len(train_data)} questions, layers: {train_layers}")

    print("Loading validation set...")
    test_data, test_layers = load_feature_activations(args.validation_pkl)
    print(f"  {len(test_data)} questions, layers: {test_layers}")

    layers = sorted(set(train_layers) & set(test_layers))
    print(f"Common layers: {layers}")

    # Load feature categories
    print("Loading feature categories...")
    categories = get_feature_categories(args.quadrant_pkl, layers)

    # Build labels
    y_train_correct = build_correctness_labels(train_data)
    y_test_correct = build_correctness_labels(test_data)
    y_train_entropy = build_entropy_labels(train_data)
    y_test_entropy = build_entropy_labels(test_data)

    print(f"Train: {sum(y_train_correct)}/{len(y_train_correct)} correct ({100*sum(y_train_correct)/len(y_train_correct):.1f}%)")
    print(f"Test:  {sum(y_test_correct)}/{len(y_test_correct)} correct ({100*sum(y_test_correct)/len(y_test_correct):.1f}%)")
    print(f"Train entropy: mean={np.mean(y_train_entropy):.4f}  std={np.std(y_train_entropy):.4f}")
    print(f"Test entropy:  mean={np.mean(y_test_entropy):.4f}  std={np.std(y_test_entropy):.4f}")

    feature_sets = ['pure_uncertainty', 'pure_incorrectness', 'both', 'all_combined']
    all_rows = []

    # =========================================================================
    # Entropy-only baseline for correctness prediction
    # =========================================================================
    print(f"\n{'='*70}")
    print("BASELINE: Predict correctness from entropy alone")
    print(f"{'='*70}")

    X_train_ent = y_train_entropy.reshape(-1, 1)
    X_test_ent = y_test_entropy.reshape(-1, 1)
    baseline_result = evaluate_classification(X_train_ent, y_train_correct, X_test_ent, y_test_correct)
    print(f"  Entropy-only baseline: AUROC={baseline_result['auroc']:.4f}  "
          f"acc={baseline_result['accuracy']:.4f}  F1={baseline_result['f1']:.4f}")

    all_rows.append({
        'target': 'correctness',
        'feature_set': 'entropy_baseline',
        'scope': 'all_layers',
        'layer': 'all',
        'n_features': 1,
        'metric1': baseline_result['auroc'],
        'metric1_name': 'auroc',
        'metric2': baseline_result['f1'],
        'metric2_name': 'f1',
        'accuracy': baseline_result['accuracy'],
    })

    # =========================================================================
    # TARGET 1: Predict correctness (classification)
    # =========================================================================
    print(f"\n{'='*70}")
    print("TARGET: correctness (logistic regression, AUROC)")
    print(f"{'='*70}")

    print(f"\nPer-layer evaluation:")
    for layer in layers:
        if layer not in categories:
            continue

        cat = categories[layer]
        layer_results = []

        for fs_name in feature_sets:
            if fs_name == 'all_combined':
                feat_indices = cat['pure_uncertainty'] + cat['pure_incorrectness'] + cat['both']
            else:
                feat_indices = cat[fs_name]

            if len(feat_indices) == 0:
                continue

            X_train = build_feature_matrix(train_data, layer, feat_indices)
            X_test = build_feature_matrix(test_data, layer, feat_indices)

            result = evaluate_classification(X_train, y_train_correct, X_test, y_test_correct)

            row = {
                'target': 'correctness',
                'feature_set': fs_name,
                'scope': f'L{layer}',
                'layer': layer,
                'n_features': result['n_features'],
                'metric1': result['auroc'],
                'metric1_name': 'auroc',
                'metric2': result['f1'],
                'metric2_name': 'f1',
                'accuracy': result['accuracy'],
            }
            all_rows.append(row)
            layer_results.append(row)

        if layer_results:
            best = max(layer_results, key=lambda r: r['metric1'])
            print(f"  L{layer:>2}: ", end="")
            for r in layer_results:
                marker = " *" if r == best else ""
                print(f"{r['feature_set'][:8]:>8}={r['metric1']:.3f}({r['n_features']}){marker}  ", end="")
            print()

    # All-layers concatenated
    print(f"\nAll-layers concatenated:")
    for fs_name in feature_sets:
        X_train_parts, X_test_parts = [], []
        for layer in layers:
            if layer not in categories:
                continue
            cat = categories[layer]
            feat_indices = (cat['pure_uncertainty'] + cat['pure_incorrectness'] + cat['both']
                           if fs_name == 'all_combined' else cat[fs_name])
            if not feat_indices:
                continue
            X_train_parts.append(build_feature_matrix(train_data, layer, feat_indices))
            X_test_parts.append(build_feature_matrix(test_data, layer, feat_indices))

        if not X_train_parts:
            continue

        X_train_cat = np.hstack(X_train_parts)
        X_test_cat = np.hstack(X_test_parts)
        result = evaluate_classification(X_train_cat, y_train_correct, X_test_cat, y_test_correct)

        print(f"  {fs_name:<20}: AUROC={result['auroc']:.4f}  acc={result['accuracy']:.4f}  "
              f"F1={result['f1']:.4f}  ({result['n_features']} features)")

        all_rows.append({
            'target': 'correctness',
            'feature_set': fs_name,
            'scope': 'all_layers',
            'layer': 'all',
            'n_features': result['n_features'],
            'metric1': result['auroc'],
            'metric1_name': 'auroc',
            'metric2': result['f1'],
            'metric2_name': 'f1',
            'accuracy': result['accuracy'],
        })

    # =========================================================================
    # TARGET 2: Predict entropy (regression)
    # =========================================================================
    print(f"\n{'='*70}")
    print("TARGET: entropy (ridge regression, Spearman r, R²)")
    print(f"{'='*70}")

    print(f"\nPer-layer evaluation:")
    for layer in layers:
        if layer not in categories:
            continue

        cat = categories[layer]
        layer_results = []

        for fs_name in feature_sets:
            if fs_name == 'all_combined':
                feat_indices = cat['pure_uncertainty'] + cat['pure_incorrectness'] + cat['both']
            else:
                feat_indices = cat[fs_name]

            if len(feat_indices) == 0:
                continue

            X_train = build_feature_matrix(train_data, layer, feat_indices)
            X_test = build_feature_matrix(test_data, layer, feat_indices)

            result = evaluate_regression(X_train, y_train_entropy, X_test, y_test_entropy)

            row = {
                'target': 'entropy',
                'feature_set': fs_name,
                'scope': f'L{layer}',
                'layer': layer,
                'n_features': result['n_features'],
                'metric1': result['spearman_r'],
                'metric1_name': 'spearman_r',
                'metric2': result['r2'],
                'metric2_name': 'r2',
                'accuracy': None,
            }
            all_rows.append(row)
            layer_results.append(row)

        if layer_results:
            best = max(layer_results, key=lambda r: r['metric1'])
            print(f"  L{layer:>2}: ", end="")
            for r in layer_results:
                marker = " *" if r == best else ""
                print(f"{r['feature_set'][:8]:>8}={r['metric1']:.3f}({r['n_features']}){marker}  ", end="")
            print()

    # All-layers concatenated
    print(f"\nAll-layers concatenated:")
    for fs_name in feature_sets:
        X_train_parts, X_test_parts = [], []
        for layer in layers:
            if layer not in categories:
                continue
            cat = categories[layer]
            feat_indices = (cat['pure_uncertainty'] + cat['pure_incorrectness'] + cat['both']
                           if fs_name == 'all_combined' else cat[fs_name])
            if not feat_indices:
                continue
            X_train_parts.append(build_feature_matrix(train_data, layer, feat_indices))
            X_test_parts.append(build_feature_matrix(test_data, layer, feat_indices))

        if not X_train_parts:
            continue

        X_train_cat = np.hstack(X_train_parts)
        X_test_cat = np.hstack(X_test_parts)
        result = evaluate_regression(X_train_cat, y_train_entropy, X_test_cat, y_test_entropy)

        print(f"  {fs_name:<20}: Spearman={result['spearman_r']:.4f}  R²={result['r2']:.4f}  "
              f"({result['n_features']} features)")

        all_rows.append({
            'target': 'entropy',
            'feature_set': fs_name,
            'scope': 'all_layers',
            'layer': 'all',
            'n_features': result['n_features'],
            'metric1': result['spearman_r'],
            'metric1_name': 'spearman_r',
            'metric2': result['r2'],
            'metric2_name': 'r2',
            'accuracy': None,
        })

    # =========================================================================
    # Save
    # =========================================================================
    csv_file = output_dir / "prediction_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {csv_file}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: All-layers concatenated")
    print(f"{'='*70}")
    print(f"{'Feature Set':<20} {'Correctness AUROC':>18} {'Entropy Spearman':>17} {'Entropy R²':>11}")
    print("-" * 70)

    # Entropy baseline first
    bl = next((r for r in all_rows if r['feature_set'] == 'entropy_baseline'), None)
    if bl:
        print(f"  {'entropy_baseline':<20} {bl['metric1']:>16.4f}{'':>17}{'':>11}")

    for fs_name in feature_sets:
        corr_row = next((r for r in all_rows if r['feature_set'] == fs_name
                        and r['scope'] == 'all_layers' and r['target'] == 'correctness'), None)
        ent_row = next((r for r in all_rows if r['feature_set'] == fs_name
                       and r['scope'] == 'all_layers' and r['target'] == 'entropy'), None)
        corr_auroc = f"{corr_row['metric1']:.4f}" if corr_row else "N/A"
        ent_spearman = f"{ent_row['metric1']:.4f}" if ent_row else "N/A"
        ent_r2 = f"{ent_row['metric2']:.4f}" if ent_row else "N/A"
        print(f"  {fs_name:<20} {corr_auroc:>16} {ent_spearman:>17} {ent_r2:>11}")

    # Best layer per feature set per target
    print(f"\n{'='*70}")
    print("Best single layer per feature set")
    print(f"{'='*70}")
    for target_name, metric_name in [('correctness', 'AUROC'), ('entropy', 'Spearman r')]:
        print(f"\n  {target_name} ({metric_name}):")
        for fs_name in feature_sets:
            layer_rows = [r for r in all_rows if r['feature_set'] == fs_name
                         and r['scope'] != 'all_layers' and r['target'] == target_name]
            if layer_rows:
                best = max(layer_rows, key=lambda r: r['metric1'])
                print(f"    {fs_name:<20}  L{best['layer']} {metric_name}={best['metric1']:.4f}  ({best['n_features']} features)")

    print(f"\nDONE")


if __name__ == '__main__':
    main()
