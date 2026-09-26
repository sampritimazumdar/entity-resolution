import pandas as pd
import numpy as np

df = pd.read_csv('output/candidate_pairs.tsv', sep='\t', dtype=str, keep_default_na=False)

# Auto-detect the candidate column
cand_col = None
for c in ['candidate_entity_ids', 'candidate_ids', 'candidates', 'source2_entity_ids']:
    if c in df.columns:
        cand_col = c
        break
if cand_col is None:
    raise SystemExit(f"No candidate column found. Columns: {list(df.columns)}")

df['n'] = df[cand_col].apply(
    lambda x: 0 if x.strip() == '' else len(x.split(',')))

print(f'Column used: {cand_col}')
print('Total S1:', len(df))
print('Empty (0 candidates):', (df['n'] == 0).sum())
print()
print('Candidate count per S1:')
print('  mean:  ', round(df['n'].mean(), 2))
print('  median:', int(df['n'].median()))
print('  p90:   ', int(np.percentile(df['n'], 90)))
print('  p99:   ', int(np.percentile(df['n'], 99)))
print('  max:   ', int(df['n'].max()))
print()
bins = [0, 1, 10, 20, 30, 100000]
labels = ['0', '1-10', '11-20', '21-30', '30+']
print('Distribution:')
for lbl, cnt in zip(labels, np.histogram(df['n'], bins=bins)[0]):
    print(f'  {lbl:>8}: {cnt}')