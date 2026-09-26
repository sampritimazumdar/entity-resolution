import pandas as pd
import sys

f1 = sys.argv[1]
f2 = sys.argv[2]
a = pd.read_csv(f1, sep='\t').set_index('source1_entity_id')
b = pd.read_csv(f2, sep='\t').set_index('source1_entity_id')
diff = a['matched_entity_ids'].fillna('') != b['matched_entity_ids'].fillna('')
print(f'Total S1: {len(a)}')
print(f'Changed between runs: {diff.sum()} ({100*diff.mean():.1f}%)')
