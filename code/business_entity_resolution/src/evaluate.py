import numpy as np
import pandas as pd

def parse_ids(s):
    if not isinstance(s, str) or s.strip() == '':
        return set()
    return set(s.split(','))

def f05_single(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred and gold:
        return 0.0
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p + r == 0:
        return 0.0
    return (1.25 * p * r) / (0.25 * p + r)

def score_predictions(pred_df, gt_df):
    """Macro-averaged F0.5 across all S1 in gt_df."""
    pred_map = dict(zip(pred_df['source1_entity_id'],
                        pred_df['matched_entity_ids'].apply(parse_ids)))
    scores = []
    for _, row in gt_df.iterrows():
        gold = parse_ids(row['matched_entity_ids'])
        pred = pred_map.get(row['source1_entity_id'], set())
        scores.append(f05_single(pred, gold))
    return float(np.mean(scores))

def score_by_country(pred_df, gt_df, s1_df):
    """Report F0.5 per country."""
    pred_map = dict(zip(pred_df['source1_entity_id'],
                        pred_df['matched_entity_ids'].apply(parse_ids)))
    country_map = dict(zip(s1_df['entity_id'], s1_df['country']))
    gt = gt_df.copy()
    gt['country'] = gt['source1_entity_id'].map(country_map)
    results = {}
    for c, group in gt.groupby('country'):
        scores = []
        for _, row in group.iterrows():
            gold = parse_ids(row['matched_entity_ids'])
            pred = pred_map.get(row['source1_entity_id'], set())
            scores.append(f05_single(pred, gold))
        results[c] = float(np.mean(scores))
    return results