"""
Selective Prediction via SAE Feature Classifiers

Use logistic regression trained on SAE feature activations to decide
whether to answer each question. Abstain when classifier confidence is low.

Reports accuracy-coverage tradeoff and compares against entropy baseline.

Usage:
    python experiments/selective_prediction.py \
        --discovery-pkl results/sae_uncertainty_mmlu_discovery_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
        --validation-pkl results/sae_uncertainty_mmlu_validation_all_layers/sae_uncertainty_Llama-3.1-8B.pkl \
        --quadrant-pkl results/sae_quadrant_mmlu_discovery_all_layers_p25/quadrant_analysis.pkl \
        --output-dir results/sae_selective_prediction_p25/
"""

import csv
import pickle
import numpy as np
from pathlib import Path
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_feature_activations(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['per_question_data'], data['config']['sae_layers']


def get_both_features(quadrant_pkl, layers):
    with open(quadrant_pkl, 'rb') as f:
        data = pickle.load(f)
    p = data['config']['p_threshold']
    features = {}
    for layer in layers:
        if layer not in data['all_comparisons']:
            continue
        comp = data['all_comparisons'][layer]
        sig_unc = {r['feature_idx'] for r in comp['pure_uncertainty']
                   if r['p_value'] < p and r['effect_size'] > 0}
        sig_inc = {r['feature_idx'] for r in comp['pure_incorrectness']
                   if r['p_value'] < p and r['effect_size'] > 0}
        both = sorted(sig_unc & sig_inc)
        if both:
            features[layer] = both
    return features


def build_matrix(data, layer, feat_indices):
    feat_to_col = {f: i for i, f in enumerate(feat_indices)}
    X = np.zeros((len(data), len(feat_indices)))
    for q_idx, q in enumerate(data):
        if layer in q['sae_features']:
            sf = q['sae_features'][layer]
            for idx, val in zip(sf['indices'], sf['values']):
                if idx in feat_to_col:
                    X[q_idx, feat_to_col[idx]] = val
    return X


def compute_selective_stats(y_true, probs, threshold):
    """Compute stats for a given confidence threshold."""
    mask = probs >= threshold
    n_answered = mask.sum()
    n_abstained = len(y_true) - n_answered
    if n_answered == 0:
        return None

    correct_answered = y_true[mask].sum()
    incorrect_answered = n_answered - correct_answered
    correct_abstained = y_true[~mask].sum()  # correct questions we didn't answer
    incorrect_abstained = n_abstained - correct_abstained

    return {
        'n_answered': int(n_answered),
        'n_abstained': int(n_abstained),
        'coverage': n_answered / len(y_true),
        'accuracy': y_true[mask].mean(),
        'correct_answered': int(correct_answered),
        'incorrect_answered': int(incorrect_answered),
        'correct_abstained': int(correct_abstained),
        'incorrect_abstained': int(incorrect_abstained),
    }


def main():
    parser = argparse.ArgumentParser(description="Selective Prediction via SAE Features")
    parser.add_argument('--discovery-pkl', type=str, required=True)
    parser.add_argument('--validation-pkl', type=str, required=True)
    parser.add_argument('--quadrant-pkl', type=str, required=True)
    parser.add_argument('--layers', type=str, default='18',
                        help='Layers to test (comma-separated, or "all" for concatenated)')
    parser.add_argument('--output-dir', type=str, default='results/sae_selective_prediction_p25/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    train_data, train_layers = load_feature_activations(args.discovery_pkl)
    test_data, test_layers = load_feature_activations(args.validation_pkl)
    both_features = get_both_features(args.quadrant_pkl, sorted(set(train_layers) & set(test_layers)))

    y_train = np.array([1 if q['correct'] else 0 for q in train_data])
    y_test = np.array([1 if q['correct'] else 0 for q in test_data])
    test_entropies = np.array([q['entropy'] for q in test_data])

    baseline_acc = y_test.mean()
    total = len(y_test)
    print(f"Baseline accuracy: {baseline_acc*100:.2f}% on {total} questions")

    # Parse layers
    if args.layers == 'all':
        test_layers_list = [('all', sorted(both_features.keys()))]
    else:
        layer_nums = [int(x) for x in args.layers.split(',')]
        test_layers_list = [(f'L{l}', [l]) for l in layer_nums]
        test_layers_list.append(('all', sorted(both_features.keys())))

    thresholds = np.arange(0.30, 0.91, 0.05)
    all_rows = []

    for layer_name, layer_list in test_layers_list:
        # Build feature matrix
        X_train_parts, X_test_parts = [], []
        n_feats = 0
        for l in layer_list:
            if l not in both_features:
                continue
            X_train_parts.append(build_matrix(train_data, l, both_features[l]))
            X_test_parts.append(build_matrix(test_data, l, both_features[l]))
            n_feats += len(both_features[l])

        if not X_train_parts:
            continue

        X_train = np.hstack(X_train_parts)
        X_test = np.hstack(X_test_parts)

        # Train classifier
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_test_s)[:, 1]

        print(f"\n{'='*80}")
        print(f"SAE Features: {layer_name} ({n_feats} features)")
        print(f"{'='*80}")
        print(f"{'Thresh':>7} {'Answered':>9} {'Abstain':>8} {'Cover':>7} {'Acc':>7} "
              f"{'Gain':>7} {'Corr_Ans':>9} {'Inc_Ans':>8} {'Corr_Abs':>9} {'Inc_Abs':>8}")
        print("-" * 85)

        for t in thresholds:
            stats = compute_selective_stats(y_test, probs, t)
            if stats is None:
                continue
            gain = (stats['accuracy'] - baseline_acc) * 100
            print(f"{t:>7.2f} {stats['n_answered']:>9} {stats['n_abstained']:>8} "
                  f"{stats['coverage']*100:>6.1f}% {stats['accuracy']*100:>6.2f}% "
                  f"{gain:>+6.2f}% {stats['correct_answered']:>9} {stats['incorrect_answered']:>8} "
                  f"{stats['correct_abstained']:>9} {stats['incorrect_abstained']:>8}")

            all_rows.append({
                'method': f'sae_{layer_name}',
                'layer': layer_name,
                'n_features': n_feats,
                'threshold': round(t, 2),
                **stats,
            })

    # Entropy baseline
    print(f"\n{'='*80}")
    print(f"Entropy Baseline")
    print(f"{'='*80}")
    print(f"{'Cover%':>7} {'Answered':>9} {'Abstain':>8} {'Cover':>7} {'Acc':>7} "
          f"{'Gain':>7} {'Corr_Ans':>9} {'Inc_Ans':>8} {'Corr_Abs':>9} {'Inc_Abs':>8}")
    print("-" * 85)

    for pct in range(90, 19, -5):
        ent_thresh = np.percentile(test_entropies, pct)
        mask = test_entropies <= ent_thresh
        n_answered = mask.sum()
        n_abstained = total - n_answered
        if n_answered == 0:
            continue

        correct_answered = int(y_test[mask].sum())
        incorrect_answered = int(n_answered - correct_answered)
        correct_abstained = int(y_test[~mask].sum())
        incorrect_abstained = int(n_abstained - correct_abstained)
        acc = y_test[mask].mean()
        coverage = n_answered / total
        gain = (acc - baseline_acc) * 100

        print(f"  p{pct:<4} {n_answered:>9} {n_abstained:>8} "
              f"{coverage*100:>6.1f}% {acc*100:>6.2f}% "
              f"{gain:>+6.2f}% {correct_answered:>9} {incorrect_answered:>8} "
              f"{correct_abstained:>9} {incorrect_abstained:>8}")

        all_rows.append({
            'method': 'entropy_baseline',
            'layer': 'output',
            'n_features': 1,
            'threshold': pct / 100,
            'n_answered': n_answered,
            'n_abstained': n_abstained,
            'coverage': coverage,
            'accuracy': acc,
            'correct_answered': correct_answered,
            'incorrect_answered': incorrect_answered,
            'correct_abstained': correct_abstained,
            'incorrect_abstained': incorrect_abstained,
        })

    # Save CSV
    csv_file = output_dir / "selective_prediction.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {csv_file}")
    print("DONE")


if __name__ == '__main__':
    main()
