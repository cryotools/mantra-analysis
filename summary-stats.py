#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:52:30 2020

@author: David Loibl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tsla_file = '<YOUR_MANTRA_FILE>.h5'

robustness_thres = 100

# Read TSLA data
print('\nReading TSLA data. This may take a while ...')
df_tsla = pd.read_hdf(tsla_file)
print('Successfully imported '+ str(df_tsla.shape[0]) +' TSLA measurements.')
df_tsla.set_index(pd.to_datetime(df_tsla['LS_DATE']), inplace=True)



# Prepare and print basic summary statistics
print('\Satellite imagery stats')
n_sat_scenes        = len(df_tsla['LS_ID'].unique())
print('Total n satellite scenes used: '+ str(n_sat_scenes))

sensors_unique      = list(df_tsla['LS_SAT'].unique())
sensors_unique.sort()

n_scenes_per_sensor = {}
for sensor in sensors_unique:
    # df_sensor_subset = df_tsla['LS_ID'][df_tsla['LS_SAT'] == sensor].unique()
    n_scenes_per_sensor[sensor] = len(df_tsla['LS_ID'][df_tsla['LS_SAT'] == sensor].unique())
    print('n scenes for '+ str(sensor) +': '+ str(n_scenes_per_sensor[sensor]))

nobs_annual = df_tsla['RGI_ID'].groupby(df_tsla.index.year).count()
print('\nAnnual n obseravations:')
print(str(nobs_annual))

print('\nPreparing histogram of n obs. per year ...')
hist_nobs_annual = nobs_annual.plot(kind='bar', title='n TSLA observations per year', xlabel='Year C.E.', ylabel='n obs.', figsize=[10,7])
plt.show()

nobs_glacier = df_tsla['RGI_ID'].groupby(df_tsla['RGI_ID']).count()

nobs_glacier_mean    = np.round(nobs_glacier.mean(), decimals=2)
nobs_glacier_median  = np.round(nobs_glacier.median(), decimals=2)
nobs_glacier_std     = np.round(nobs_glacier.std(), decimals=2)
nobs_glacier_mad     = np.round(nobs_glacier.mad(), decimals=2)
nobs_glacier_min     = np.round(nobs_glacier.min(), decimals=2)
nobs_glacier_max     = np.round(nobs_glacier.max(), decimals=2)

n_glaciers_gt_nobsthres = np.round(nobs_glacier[nobs_glacier > robustness_thres].count(), decimals=2)

print('Mean n obs. per glacier:   '+ str(nobs_glacier_mean) +' +/- '+ str(nobs_glacier_std) +'(1 sigma)')
print('Median n obs. per glacier: '+ str(nobs_glacier_median) +' +/- '+ str(nobs_glacier_mad) +'(1 MAD)')

print('\n'+ str(n_glaciers_gt_nobsthres) +' glaciers fullfil the minimum robustness criterium of '+ str(robustness_thres) +' n obs.')

print('Preparing histogram of n obs. per glacier ...')
hist_nobs_glacier = nobs_glacier.hist(bins=100, figsize=[10,7]) #bins=100, title='n TSLA observations per glacier', xlabel='n obs. per glacier', ylabel='n glaciers', )
plt.show()

tsla_mean           = np.round(df_tsla['SC_median'].mean(), decimals=2)
tsla_median         = np.round(df_tsla['SC_median'].median(), decimals=2)
tsla_std            = np.round(df_tsla['SC_median'].std(), decimals=2)
tsla_mad            = np.round(df_tsla['SC_median'].mad(), decimals=2)
tsla_mode           = np.round(float(df_tsla['SC_median'].mode()), decimals=2)
tsla_99perc         = df_tsla['SC_median'].quantile([0.005, 0.995])
tsla_value_cnts_20  = df_tsla['SC_median'].value_counts(bins=20)

print('\nMean TSLA:   '+ str(tsla_mean) +' +/- '+ str(tsla_std) +'(1 sigma) m a.s.l.')
print('Median TSLA: '+ str(tsla_median) +' +/- '+ str(tsla_mad) +'(1 MAD) m a.s.l.')
print('Mode TSLA:   '+ str(tsla_mode) +' m a.s.l.')

print('\n99% of the TSLA values are in a range between: '+ str(list(tsla_99perc)[0]) +' and '+ str(list(tsla_99perc)[1]) +' m a.s.l.')

print('\nPreparing histogram for TSLA values (altitude bins) ...')
hist_tsla = df_tsla['SC_median'].hist(bins=100, figsize=[10,7]) #, title='TSLA histogram', xlabel='TSLA (m a.s.l.)', ylabel='n'
plt.show()

print('\n Uncertainty metrics')
stdev_mean           = np.round(df_tsla['SC_stdev'].mean(), decimals=2)
stdev_median         = np.round(df_tsla['SC_stdev'].median(), decimals=2)
stdev_std            = np.round(df_tsla['SC_stdev'].std(), decimals=2)
stdev_mad            = np.round(df_tsla['SC_stdev'].mad(), decimals=2)
stdev_max            = np.round(df_tsla['SC_stdev'].max(), decimals=2)
stdev_min            = np.round(df_tsla['SC_stdev'].min(), decimals=2)

print('\nMean Std. Dev. for TSLA measurements:   '+ str(stdev_mean) +' +/- '+ str(stdev_std) +'(1 sigma) m')
print('Median Std. Dev. for TSLA measurements: '+ str(stdev_median) +' +/- '+ str(stdev_mad) +'(1 MAD) m')
