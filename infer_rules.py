import pandas as pd
import numpy as np
import re
import os
import time
import gc
from unidecode import unidecode
from collections import defaultdict
from rapidfuzz import fuzz

TOP_K = 25
NAME_THR = 0.82
ADDR_THR = 0.65
CHUNK = 500_000

def norm(s):
    if not isinstance(s, str):
        return ''
    s = unidecode(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def make_key(name, country):
    tokens = norm(name).split()[:2]
    return str(country) + '|' + ' '.join(tokens)

def stream_tsv(path, usecols=None):
    for chunk in pd.read_csv(path, sep='\t', chunksize=CHUNK,
                             usecols=usecols, dtype=str):
        yield chunk

t0 = time.time()
os.makedirs('output', exist_ok=True)

print("[1/3] Building blocking index from S2 + S3 (chunked) ...")
index = defaultdict(list)
pool_nname = {}
pool_naddr = {}

for src in ['dataset/test/test_source2.tsv', 'dataset/test/test_source3.tsv']:
    print(f"    reading {src}")
    for chunk in stream_tsv(src, usecols=['entity_id','business_name','business_address','country']):
        for eid, bn, ba, co in zip(chunk['entity_id'], chunk['business_name'],
                                    chunk['business_address'], chunk['country']):
            k = make_key(bn, co)
            index[k].append(eid)
            pool_nname[eid] = norm(bn)
            pool_naddr[eid] = norm(ba)
        del chunk
        gc.collect()
print(f"    index: {len(index)} keys, {len(pool_nname)} records ({time.time()-t0:.1f}s)")

print("[2/3] Scoring S1 in chunks ...")
out_m = open('output/matching_results.tsv', 'w', encoding='utf-8')
out_c = open('output/candidate_pairs.tsv', 'w', encoding='utf-8')
out_m.write("source1_entity_id\tmatched_entity_ids\n")
out_c.write("source1_entity_id\tcandidate_entity_ids\n")

n_total = 0
n_matched = 0
for chunk in stream_tsv('dataset/test/test_source1.tsv',
                        usecols=['entity_id','business_name','business_address','country']):
    for sid, bn, ba, co in zip(chunk['entity_id'], chunk['business_name'],
                                chunk['business_address'], chunk['country']):
        n_total += 1
        if n_total % 100000 == 0:
            print(f"    {n_total} S1 processed ({time.time()-t0:.1f}s)")

        nname = norm(bn)
        naddr = norm(ba)
        key = str(co) + '|' + ' '.join(nname.split()[:2])
        cands = index.get(key, [])

        if not cands:
            out_m.write(f"{sid}\t\n")
            out_c.write(f"{sid}\t\n")
            continue

        scored = []
        for cid in cands:
            ns = fuzz.token_set_ratio(nname, pool_nname.get(cid, '')) / 100
            scored.append((cid, ns))
        scored.sort(key=lambda x: -x[1])
        top = scored[:TOP_K]

        kept = []
        for cid, ns in top:
            if ns < NAME_THR:
                continue
            as_ = fuzz.token_set_ratio(naddr, pool_naddr.get(cid, '')) / 100
            if as_ >= ADDR_THR:
                kept.append(cid)

        if kept:
            n_matched += 1
        out_m.write(f"{sid}\t{','.join(kept)}\n")
        out_c.write(f"{sid}\t{','.join(c[0] for c in top)}\n")

    del chunk
    gc.collect()

out_m.close()
out_c.close()
print(f"[3/3] DONE in {time.time()-t0:.1f}s")
print(f"    S1 total: {n_total}")
print(f"    S1 with >=1 match: {n_matched}")
print(f"    Singletons (empty): {n_total - n_matched}")
