import pandas as pd
import numpy as np
import re
import time
import joblib
from unidecode import unidecode
from collections import defaultdict
from rapidfuzz import fuzz
from sklearn.ensemble import GradientBoostingClassifier

CHUNK = 500_000
SAMPLE_N = 8000
TOP_K = 30

def norm(s):
    if not isinstance(s, str):
        return ''
    s = unidecode(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def make_key(name, country):
    tokens = norm(name).split()[:2]
    return str(country) + '|' + ' '.join(tokens)

def featurize(s1_name, s1_addr, c_name, c_addr, same_country):
    return {
        'name_token_set': fuzz.token_set_ratio(s1_name, c_name) / 100,
        'name_ratio':     fuzz.ratio(s1_name, c_name) / 100,
        'name_partial':   fuzz.partial_ratio(s1_name, c_name) / 100,
        'addr_token_set': fuzz.token_set_ratio(s1_addr, c_addr) / 100,
        'addr_ratio':     fuzz.ratio(s1_addr, c_addr) / 100,
        'addr_partial':   fuzz.partial_ratio(s1_addr, c_addr) / 100,
        'same_country':   int(same_country),
    }

t0 = time.time()
print("[1/5] Loading train data...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=SAMPLE_N)
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t',
                 usecols=['entity_id','business_name','business_address','country'])
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t',
                 usecols=['entity_id','business_name','business_address','country'])
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
print(f"  loaded ({time.time()-t0:.1f}s)")

print("[2/5] Building index...")
pool = pd.concat([s2, s3], ignore_index=True)
pool['nname'] = pool['business_name'].fillna('').map(norm)
pool['naddr'] = pool['business_address'].fillna('').map(norm)
index = defaultdict(list)
for eid, bn, co in zip(pool['entity_id'], pool['business_name'], pool['country']):
    index[make_key(bn, co)].append(eid)
pool_lookup = pool.set_index('entity_id')
print(f"  index: {len(index)} keys, {len(pool)} rows ({time.time()-t0:.1f}s)")

print("[3/5] Building feature matrix...")
gt_map = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids'].fillna('')))
rows = []
for i, r in s1.iterrows():
    if i % 20000 == 0:
        print(f"    {i}/{len(s1)} ({time.time()-t0:.1f}s)")
    sid = r['entity_id']
    nname = norm(r['business_name'])
    naddr = norm(r['business_address'])
    key = make_key(r['business_name'], r['country'])
    cands = index.get(key, [])
    if not cands:
        continue
    gold = set(str(gt_map.get(sid, '')).split(',')) - {''}
    scored = []
    for cid in cands:
        c = pool_lookup.loc[cid]
        ns = fuzz.token_set_ratio(nname, c['nname']) / 100
        scored.append((cid, ns))
    scored.sort(key=lambda x: -x[1])
    for cid, _ in scored[:TOP_K]:
        c = pool_lookup.loc[cid]
        feat = featurize(nname, naddr, c['nname'], c['naddr'],
                         r['country'] == c['country'])
        feat['label'] = int(cid in gold)
        feat['s1'] = sid
        feat['cid'] = cid
        rows.append(feat)
print(f"  {len(rows)} pairs ({time.time()-t0:.1f}s)")

print("[4/5] Training classifier...")
df = pd.DataFrame(rows)
feat_cols = ['name_token_set','name_ratio','name_partial',
             'addr_token_set','addr_ratio','addr_partial','same_country']
X = df[feat_cols].values
y = df['label'].values
print(f"  positives: {y.sum()}, negatives: {len(y)-y.sum()}")

s1_ids = df['s1'].unique()
rng = np.random.default_rng(42)
rng.shuffle(s1_ids)
val_ids = set(s1_ids[:int(0.2*len(s1_ids))])
tr = df[~df['s1'].isin(val_ids)]
va = df[df['s1'].isin(val_ids)]

clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(tr[feat_cols], tr['label'])

print("[5/5] Threshold sweep...")
va = va.copy()
va['prob'] = clf.predict_proba(va[feat_cols])[:, 1]

best_t, best_f = 0.5, 0.0
for t in np.arange(0.3, 0.95, 0.05):
    per_s1 = []
    for sid, g in va.groupby('s1'):
        pred = set(g[g['prob'] >= t]['cid'])
        gold = set(g[g['label'] == 1]['cid'])
        if not pred and not gold:
            per_s1.append(1.0)
        elif not pred or not gold:
            per_s1.append(0.0)
        else:
            tp = len(pred & gold); fp = len(pred - gold); fn = len(gold - pred)
            p = tp/(tp+fp) if (tp+fp) else 0
            r = tp/(tp+fn) if (tp+fn) else 0
            f = (1.25*p*r)/(0.25*p+r) if (p+r) else 0
            per_s1.append(f)
    f05 = np.mean(per_s1)
    print(f"  t={t:.2f}  F0.5={f05:.4f}")
    if f05 > best_f:
        best_f, best_t = f05, t

print(f"BEST: t={best_t:.2f}  F0.5={best_f:.4f}")
joblib.dump({'model': clf, 'threshold': best_t, 'features': feat_cols},
            'code/business_entity_resolution/model.joblib')
print("Saved model.joblib")

imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(imp.to_string())
