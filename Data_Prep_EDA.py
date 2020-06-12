import sys
sys.path.insert(0, './functions.py')

from functions import *

import pandas as pd

### Prepping the data ###
train_df = pd.read_csv('./data/interview_data.csv')

domain_dummies           = get_top_domain_dummies(train_df)
from_domain_hash_dummies = get_top_from_domain_dummies(train_df)
dow_dummies              = get_dow_dummies(train_df)

model_df = pd.concat([train_df, dow_dummies, domain_dummies, from_domain_hash_dummies], axis = 1)
model_df['read_rate_discrete_even'] = pd.cut(train_df.read_rate, 3, labels = ['Low', 'Med', 'High'])
model_df['read_rate_discrete'] = model_df['read_rate'].map(read_rate_qcut)
model_df.drop(columns = ['id','from_domain_hash','Domain_extension', 'day'], inplace = True)

model_df.to_csv('./data/interview_data_prepped.csv', index = False)

### Generating EDA Figures ###
make_corr_graph(model_df, 'correlations_bargraph.png')
make_histogram(model_df.read_rate, 'read_rate_hist.png')
make_histogram(model_df.loc[model_df['read_rate'] > 0, 'read_rate'], 'read_rate_hist_no_zero.png')

print("--High/Med/Low bin ranges--\n      (evenly spaced)")
print(pd.cut(train_df.read_rate, 3).value_counts().sort_index())

print("\n--High/Med/Low bins--\n  (defined by Quantile ranges)")
print(pd.qcut(train_df.read_rate, 4, duplicates='drop').value_counts().sort_index())

print("\n--High/Med/Low bins--\n  (Quantile ranges, removed 0)")
print(pd.qcut(train_df.loc[train_df.read_rate > 0, 'read_rate'], 3).value_counts().sort_index())

### Printing Baseline accuracy using Quantile Ranges with Zero removed
print("\nBaseline Accuracy using Q-Bins w/o Zero:")
print(model_df.read_rate_discrete.value_counts(normalize = True))
