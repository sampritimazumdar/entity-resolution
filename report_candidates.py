"""Report candidate-pair stats for output/candidate_pairs.tsv."""
import argparse
import numpy as np
import pandas as pd

CAND_COL = 'candidate_ids'


def parse_ids(s):
    if not isinstance(s, str) or not s.strip():
        return set()
    return {x.strip() for x in s.split(',') if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='output/candidate_pairs.tsv')
    ap.add_argument('--col',  default=CAND_COL)
    args = ap.parse_args()

    df = pd.read_csv(args.path, sep='\t')
    if args.col not in df.columns:
        raise SystemExit(
            f"Column '{args.col}' not found. Columns are: {list(df.columns)}"
        )

    counts = df[args.col].apply(lambda s: len(parse_ids(s)))
    n = len(counts)

    print(f'=== {args.path} ===')
    print(f'Total S1: {n}')
    print()
    print(f'Avg candidates/S1 : {counts.mean():.2f}')
    print(f'p50               : {np.percentile(counts, 50):.0f}')
    print(f'p90               : {np.percentile(counts, 90):.0f}')
    print(f'max               : {counts.max()}')
    print()

    bins   = [-1, 0, 10, 20, 30, 10**9]
    labels = ['0', '1-10', '11-20', '21-30', '30+']
    buckets = pd.cut(counts, bins=bins, labels=labels)
    print('Distribution:')
    for lab in labels:
        c = int((buckets == lab).sum())
        pct = 100 * c / n if n else 0
        print(f'  {lab:>6s} : {c:>7d}  ({pct:5.1f}%)')


if __name__ == '__main__':
    main()