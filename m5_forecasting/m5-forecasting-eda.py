#!/usr/bin/env python
# coding: utf-8

# # # **EXPLORATORY DATA ANALYSIS FOR M5**

# # ## **INITIALIZATION**

import sys
# #print(sys.version)

# # load required packages
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pylab as pl

import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew

import gc
import lightgbm as lgb

# # ignore warnings from sklearn and seaborn
import warnings
def ignore_warn(*args, **kwargs):
     pass
warnings.warn = ignore_warn

# # pandas output format
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
pd.options.display.max_columns = 50

# # check files available
# from subprocess import check_output
# #print(check_output(['ls', os.getcwd()]).decode('utf8'))


# # ## **EXPLORATION**

cal_dtypes = {'event_name_1': 'category', 'event_name_2': 'category', 
              'event_type_1': 'category', 'event_type_2': 'category',
              'weekday': 'category', 'wm_yr_wk': 'int16', 'wday': 'int16',
              'month': 'int16', 'year': 'int16', 'snap_CA': 'float32', 
              'snap_TX': 'float32', 'snap_WI': 'float32'}
price_dtypes = {'store_id': 'category', 'item_id': 'category', 'wm_yr_wk': 'int16',
               'sell_price': 'float32'}

# # parameters for constructing time series
h = 28 # forecast horizon
max_lags = 57
tr_last = 1913 # last training observation
fday = datetime(2016, 4, 25) # forecast start date
fday

# # construct time series
def create_df(is_train = True, nrows = None, first_day = 1200):
    prices = pd.read_csv('sell_prices.csv', dtype = price_dtypes)
    for col, col_dtype in price_dtypes.items():
        if col_dtype == 'category':
            prices[col] = prices[col].cat.codes.astype('int16')
            prices[col] -= prices[col].min() # scaling
    cal = pd.read_csv('calendar.csv', dtype = cal_dtypes)
    cal['date'] = pd.to_datetime(cal['date'])
    for col, col_dtype in cal_dtypes.items():
        if col_dtype == 'category':
            cal[col] = cal[col].cat.codes.astype('int16')
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train else tr_last - max_lags, first_day)
    numcols = [f'd_{day}' for day in range(start_day, tr_last+1)] #sales data rolling window
    catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    dtype = {numcol: 'float32' for numcol in numcols}
    dtype.update({col: 'category' for col in catcols if col != 'id'})
    df = pd.read_csv('sales_train_validation.csv', nrows = nrows, 
                     usecols = catcols + numcols, dtype = dtype)
    for col in catcols:
        if col != 'id':
            df[col] = df[col].cat.codes.astype('int16')
            df[col] -= df[col].min()
    if not is_train:
        for day in range(tr_last + 1, tr_last + 28 + 1):
            df[f'd_{day}'] = np.nan
    df = pd.melt(df, 
                 id_vars = catcols,
                 value_vars = [col for col in df.columns if col.startswith('d_')], # numeric
                 var_name = 'd', # day
                 value_name = 'sales')
    df = df.merge(cal, on='d', copy = False)
    df = df.merge(prices, on = ['store_id', 'item_id', 'wm_yr_wk'], copy=False)
    return df 

# # create forecast series
def create_fea(df):
     lags = [7, 28]
     lag_cols = [f'lag_{lag}' for lag in lags]
     for lag, lag_col in zip(lags, lag_cols):
         df[lag_col] = df[['id', 'sales']].groupby('id')['sales'].shift(lag)
        
     wins = [7, 28] # windows
     for win in wins:
         for lag, lag_col in zip(lags, lag_cols):
             df[f'rmean_{lag}_{win}'] = df[['id', lag_col]].groupby('id')[lag_col].transform(lambda x: x.rolling(win).mean())
    
     date_features = {
         'wday': 'weekday',
         'week': 'weekofyear',
         'month': 'month',
         'quarter': 'quarter',
         'year': 'year',
         'mday': 'day'}
    
     for date_feat_name, date_feat_func in date_features.items():
         if date_feat_name in df.columns:
             df[date_feat_name] = df[date_feat_name].astype('int16')
         else:
             df[date_feat_name] = getattr(df['date'].dt, date_feat_func).astype('int16')


df = create_df(is_train=True, first_day = 1000) #skip days to save on memory\ndf.shape')
create_fea(df)

# # drop nans
df.dropna(inplace=True)

# # model
cat_feats = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + ['event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']
useless_cols = ['id', 'date', 'sales', 'd', 'wm_yr_wk', 'weekday']
train_cols = df.columns[~df.columns.isin(useless_cols)]
X_train = df[train_cols]
y_train = df['sales']

# np.random.seed(777)
# fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace=False)
# train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
# train_data = lgb.Dataset(X_train.loc[train_inds], label = y_train.loc[train_inds], categorical_feature = cat_feats, free_raw_data=False)
# fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds], categorical_feature = cat_feats, free_raw_data=False)

# del df, X_train, y_train, fake_valid_inds, train_inds
# gc.collect()

# params = {
#     'objective' : 'poisson',
#     'metric' : 'rmse',
#     'force_row_wise': True,
#     'learning_rate': 0.075,
#     'sub_row': 0.75, 
#     'baggin_freq': 1,
#     'lambda_l2': 0.1,
#     'metric': ['rmse'],
#     'verbosity': 1,
#     'num_iterations': 1200,
#     'num_leaves': 128,
#     'min_data_in_leaf': 100
# }

# m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20)

# plt.rcParams['figure.figsize'] = (18,4)
# fig, ax = plt.subplots(figsize=(12,8))
# lgb.plot_importance(m_lgb, max_num_features=50, height=0.8, ax=ax)
# ax.grid(False)
# plt.title('LightGBM - Feature Importance', fontsize=15)
# plt.show()

# m_lgb.save_model('model.lgb')
# #m_lgb.save_model('lgb_m5.txt', num_iteration = model.best_iteration)
m_lgb = lgb.Booster(model_file='model.lgb')

# ## **PREDICTION**

def create_lag_features_for_test(df, day):
    lags = [7, 28]
    lag_cols = [f'lag_{lag}' for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df.loc[df.date == day, lag_col] = df.loc[df.date == day-timedelta(days=lag), 'sales'].values

    windows = [7,28]
    for window in windows:
        for lag in lags:
            df_window = df[(df.date <= day-timedelta(days=lag)) 
                           & (df.date > day-timedelta(days=lag+window))]
            df_window_grouped = df_window.groupby('id').agg({'sales':'mean'}).reindex(df.loc[df.date==day,'id'])
            df.loc[df.date == day, f'rmean_{lag}_{window}'] = df_window_grouped.sales.values

def create_date_features_for_test(df):
    date_features = {
        'wday': 'weekday',
        'week': 'weekofyear',
        'month': 'month',
        'quarter': 'quarter',
        'year': 'year',
        'mday': 'day'}
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in df.columns:
            df[date_feat_name] = df[date_feat_name].astype('int16')
        else:
            df[date_feat_name] = getattr(df['date'].dt, date_feat_func).astype('int16')

alphas = [1.028, 1.023, 1.018]
weights = [1/len(alphas)]*len(alphas)
sub = 0.

te0 = create_df(False)
create_date_features_for_test(te0)
for icount, (alpha, weight) in enumerate(zip(alphas, weights)):
	te = te0.copy()
	cols =[f'F{i}' for i in range(1, 29)]
	for tdelta in range(0, 28):
		day = fday + timedelta(days=tdelta)
		print(tdelta, day.date())

		tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()

		create_lag_features_for_test(tst, day)
		tst = tst.loc[tst.date == day, train_cols]
		te.loc[te.date == day, 'sales'] = alpha * m_lgb.predict(tst) # magic multiplier by kyakovlev\n    \n    te_sub = te.loc[te.date >= fday, ['id', 'sales']].copy()\n    te_sub['F'] = [f'F{rank}' for rank in te_sub.groupby('id')['id'].cumcount()+1]\n    te_sub = te_sub.set_index(['id', 'F']).unstack()['sales'][cols].reset_index()\n    te_sub.fillna(0., inplace=True)\n    te_sub.sort_values('id', inplace=True)\n    te_sub.reset_index(drop=True, inplace=True)\n    te_sub.to_csv(f'submission_{icount}.csv', index=False)\n    if icount == 0:\n        sub = te_sub\n        sub[cols] *= weight\n    else:\n        sub[cols] += te_sub[cols]*weight\n    print(icount, alpha, weight)")

	te_sub = te.loc[te.date >= fday, ['id', 'sales']].copy()
	te_sub['F'] = [f'F{rank}' for rank in te_sub.groupby('id')['id'].cumcount()+1]
	te_sub = te_sub.set_index(['id', 'F']).unstack()['sales'][cols].reset_index()
	te_sub.fillna(0., inplace=True)
	te_sub.sort_values('id', inplace=True)
	te_sub.reset_index(drop=True, inplace=True)
	te_sub.to_csv(f'submission_{icount}.csv', index=False)
	if icount == 0:
		sub = te_sub
		sub[cols] *= weight
	else:
		sub[cols] += te_sub[cols]*weight
	print(icount, alpha, weight)

sub.id.nunique(), sub['id'].str.contains('validation$').sum()

sub2 = sub.copy()
sub2['id'] = sub2['id'].str.replace('validation$', 'evaluation')
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv('submission_lgb.csv', index=False)