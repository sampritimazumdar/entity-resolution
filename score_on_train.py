import sys
sys.path.insert(0, 'code/business_entity_resolution')
import pandas as pd
from src.evaluate import score_predictions, score_by_country

pred = pd.read_csv('output/matching_results.tsv', sep='\t')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', usecols=['entity_id','country'])

# Note: only run this AFTER we produce a matching_results on train data
print('Overall F0.5:', score_predictions(pred, gt))
print('By country:', score_by_country(pred, gt, s1))