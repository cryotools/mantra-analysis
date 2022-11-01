#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make histograms of number of observations 
  (a) per year and
  (b) by elevation bins.

@author: David Loibl
"""

import pandas as pd
# import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

#import sys
#import os
#import re


# Path setup
input_path_rgi     = './data/RGI/'
input_path_tslprep = './data/MANTRA/preprocessed/'
output_path_plots  = '<OUTPUT_PATH_PLOTS>'
output_path_tables = '<OUTPUT_PATH_PLOTS>'



# Configuration
use_fit_regions    = True
rgi_filename       = 'rgi60_Asia.csv' 

input_file = '<YOUR_MANTRA_FILE>.h5'
n_elevation_bins   = 100

dpi                = 300       # Plot resolution


# Import RGI data
df_rgi = pd.read_csv(input_path_rgi + rgi_filename, index_col='RGIId', parse_dates=True, low_memory=False)
df_rgi['RGI_ID'] = df_rgi.index
df_rgi.head()

# Import TSLA data
print('\nReading TSL input file. This may take a while ...')
df_tsl = pd.read_hdf(input_path_tslprep + input_file, parse_dates=True, index_col='LS_DATE', low_memory=False)
print('Success. Data frame contains '+ str(df_tsl.shape[0]) +' rows and '+ str(df_tsl.shape[1]) +' columns. \n')
df_tsl.index = pd.to_datetime(df_tsl['LS_DATE'])
#df_tsl.head()

n_obs_orig = df_tsl.shape[0]
n_glaciers_orig = len(df_tsl['RGI_ID'].unique())

print('Preparing time series histogram with annual bins ...')
# Merge dataframe with RGI data
df_tsl_rgi = pd.merge(df_tsl, df_rgi, how='left', on='RGI_ID')
df_tsl_rgi.index = pd.to_datetime(df_tsl_rgi['LS_DATE'])

n_obs_per_year = df_tsl['tool_version'].groupby(df_tsl.index.year).count()

n_obs_per_year.plot.bar(figsize=(16, 2))
fig = plt.gcf()
fig.savefig(path_plots +'n_obs_per_year_'+ str(dpi) +'dpi.png', dpi=dpi)


print('Preparing elevation histogram with '+ str(n_elevation_bins) +' bins ...')
fig.clf()
df_tsl.drop(columns=['LS_DATE'], inplace=True)
df_tsl.reset_index(inplace=True)

df_tsl['SC_median'].hist(bins=n_elevation_bins, grid=False, figsize=(16, 3))
fig = plt.gcf()
fig.savefig(path_plots +'n_obs_elev_hist_'+ str(dpi) +'dpi.png', dpi=dpi)



print('\nProcessing finished.\n')



