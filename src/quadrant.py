import pickle
import numpy as np
from scipy import stats


def build_activation_matrix(questions, layer_idx, active_features, feat_to_col):
    n_active = len(active_features)
    acts = np.zeros((len(questions), n_active))
    for i, d in enumerate(questions):
        if layer_idx in d['sae_features']:
            sf = d['sae_features'][layer_idx]
            for idx, val in zip(sf['indices'], sf['values']):
                if idx in feat_to_col:
                    acts[i, feat_to_col[idx]] = val
    return acts


def differential_analysis(group1_acts, group2_acts, active_features, min_freq=0.01):
    """Compare two groups. Positive effect = more active in group2."""
    results = []
    for col_idx, feat_idx in enumerate(active_features):
        g1 = group1_acts[:, col_idx]
        g2 = group2_acts[:, col_idx]

        g1_freq = (g1 > 0).mean()
        g2_freq = (g2 > 0).mean()

        if g1_freq < min_freq and g2_freq < min_freq:
            continue

        g1_mean = g1.mean()
        g2_mean = g2.mean()

        try:
            _, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        except ValueError:
            p_value = 1.0

        pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        effect_size = (g2_mean - g1_mean) / pooled_std if pooled_std > 0 else 0.0

        results.append({
            'feature_idx': feat_idx,
            'group1_mean': float(g1_mean),
            'group2_mean': float(g2_mean),
            'group1_freq': float(g1_freq),
            'group2_freq': float(g2_freq),
            'effect_size': float(effect_size),
            'p_value': float(p_value),
        })

    results.sort(key=lambda x: abs(x['effect_size']), reverse=True)
    return results


def load_quadrant_features(quadrant_pkl, layers, top_n=5, both_ranking='unc'):
    """Load features per category from quadrant analysis pickle."""
    with open(quadrant_pkl, 'rb') as f:
        data = pickle.load(f)

    p_threshold = data['config']['p_threshold']
    features_by_category = {}

    for layer in layers:
        if layer not in data['all_comparisons']:
            print(f"  WARNING: Layer {layer} not in quadrant analysis, skipping")
            continue

        comp = data['all_comparisons'][layer]

        sig_pure_unc = {r['feature_idx'] for r in comp['pure_uncertainty']
                       if r['p_value'] < p_threshold and r['effect_size'] > 0}
        sig_pure_inc = {r['feature_idx'] for r in comp['pure_incorrectness']
                       if r['p_value'] < p_threshold and r['effect_size'] > 0}

        pure_unc_only = sig_pure_unc - sig_pure_inc
        pure_inc_only = sig_pure_inc - sig_pure_unc
        both = sig_pure_unc & sig_pure_inc

        unc_ranked = sorted(
            [r for r in comp['pure_uncertainty'] if r['feature_idx'] in pure_unc_only],
            key=lambda x: x['effect_size'], reverse=True
        )
        inc_ranked = sorted(
            [r for r in comp['pure_incorrectness'] if r['feature_idx'] in pure_inc_only],
            key=lambda x: x['effect_size'], reverse=True
        )
        inc_effect_lookup = {r['feature_idx']: r['effect_size']
                             for r in comp['pure_incorrectness']}
        if both_ranking == 'min':
            both_key = lambda x: min(x['effect_size'], inc_effect_lookup.get(x['feature_idx'], 0))
        else:
            both_key = lambda x: x['effect_size']
        both_ranked = sorted(
            [r for r in comp['pure_uncertainty'] if r['feature_idx'] in both],
            key=both_key, reverse=True
        )

        features_by_category[layer] = {
            'pure_uncertainty': [r['feature_idx'] for r in unc_ranked[:top_n]],
            'pure_incorrectness': [r['feature_idx'] for r in inc_ranked[:top_n]],
            'both': [r['feature_idx'] for r in both_ranked[:top_n]],
        }

        for cat, feats in features_by_category[layer].items():
            print(f"  Layer {layer} [{cat}]: {feats} ({len(feats)} features)")

    return features_by_category


def load_both_features(quadrant_pkl, layers):
    """Load all 'both' category features (no top-N limit), ranked by min(unc, inc) effect."""
    with open(quadrant_pkl, 'rb') as f:
        data = pickle.load(f)

    p_threshold = data['config']['p_threshold']
    both_features = {}

    print(f"  p_threshold={p_threshold}")
    for layer in layers:
        if layer not in data['all_comparisons']:
            print(f"  WARNING: Layer {layer} not in quadrant analysis, skipping")
            continue

        comp = data['all_comparisons'][layer]

        sig_pure_unc = {r['feature_idx'] for r in comp['pure_uncertainty']
                       if r['p_value'] < p_threshold and r['effect_size'] > 0}
        sig_pure_inc = {r['feature_idx'] for r in comp['pure_incorrectness']
                       if r['p_value'] < p_threshold and r['effect_size'] > 0}

        both = sig_pure_unc & sig_pure_inc

        inc_effect_lookup = {r['feature_idx']: r['effect_size']
                             for r in comp['pure_incorrectness']}
        both_ranked = sorted(
            [r for r in comp['pure_uncertainty'] if r['feature_idx'] in both],
            key=lambda x: min(x['effect_size'], inc_effect_lookup.get(x['feature_idx'], 0)),
            reverse=True
        )

        both_features[layer] = [(r['feature_idx'], r['effect_size'],
                                  inc_effect_lookup.get(r['feature_idx'], 0))
                                 for r in both_ranked]

        print(f"  Layer {layer}: {len(both_features[layer])} 'both' features: "
              f"{[f[0] for f in both_features[layer]]}")

    return both_features
