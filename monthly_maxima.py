#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:04:42 2019

Helper tool to convert a full TSL file into individual files per glacier

@author: David Loibl
"""

import pandas as pd
import numpy as np
import calendar as cal


# Input files and pathes
input_file         = '<YOUR_MANTRA_FILE>.h5'
rgi_file           = 'rgi60_Asia.csv'
region_file        = 'GTN-G-glacier-subregions.csv'

input_path_rgi     = './data/RGI/'
input_path_tslprep = './data/MANTRA/preprocessed/'
output_path_plots  = '<OUTPUT_PATH_PLOTS>'
output_path_tables = '<OUTPUT_PATH_PLOTS>'

# Import RGI data
df_rgi = pd.read_csv(input_path_rgi + rgi_file, index_col='RGIId', parse_dates=True, low_memory=False)
# df_rgi_sorted = df_rgi.sort_values(by='Area', ascending=False)
df_rgi['RGI_ID'] = df_rgi.index
df_rgi.head()

# Import

print('\nReading input file. This may take a while ...')
df_tsl = pd.read_hdf(input_path_tslprep + input_file, parse_dates=True, index_col='LS_DATE', low_memory=False)
print('Success. Data frame contains '+ str(df_tsl.shape[0]) +' rows and '+ str(df_tsl.shape[1]) +' columns. \n')
#df_tsl.head()

n_obs_orig = df_tsl.shape[0]
n_glaciers_orig = len(df_tsl['RGI_ID'].unique())

# Merge dataframe with RGI data
df_tsl_rgi = pd.merge(df_tsl, df_rgi, how='left', on='RGI_ID')
df_tsl_rgi.index = pd.to_datetime(df_tsl_rgi['LS_DATE'])

# print('Extracting accumulation phase subset.')
# ablphase_subset = np.where((df_tsl_rgi.month >= 7) & (df_tsl_rgi.month <= 10))[0]
# df_tsl_rgi = df_tsl_rgi.iloc[ablphase_subset,:]
#n_obs_abl = df_tsl_rgi.shape[0]
#n_obs_diff = n_obs_orig - n_obs_abl
# print('Removed '+ str(n_obs_diff) +' accumulation phase observations ('+ str(n_obs_abl) +' remaining).')

'''
# Optionally, filter by an area threshold
area_threshold = 2
print('Extracting subset of glacier with minimum area of '+ str(area_threshold) +' km² ...')
robust_size_subset = np.where(df_tsl_rgi['glacier_area'] >= area_threshold)[0] 
df_tsl_rgi = df_tsl_rgi.iloc[robust_size_subset,:]
n_glaciers_robust = len(df_tsl_rgi['RGI_ID'].unique())
n_glaciers_diff = n_glaciers_orig - n_glaciers_robust
print('Removed '+ str(n_glaciers_diff) +' glaciers < '+ str(area_threshold) +' km² ('+ str(n_glaciers_robust) +' remaining).')
'''

'''
# Optionally, filter by a threshold of n observation per glacier
n_obs_threshold = 300
df_tsl_rgi['n_obs_per_glacier'] = df_tsl_rgi.groupby(['RGI_ID'])['RGI_ID'].transform('count')
#df_n_obs = df_tsl_rgi[['RGI_ID', 'glacier_area']].groupby(['RGI_ID']).agg(['count'])
#df_n_obs = df_n_obs.glacier_area['count']

print('Extracting subset of glacier with minimum '+ str(n_obs_threshold) +' TSL measurements ...')
robust_size_subset = np.where(df_tsl_rgi['n_obs_per_glacier'] >= n_obs_threshold)[0] 
df_tsl_rgi = df_tsl_rgi.iloc[robust_size_subset,:]
n_glaciers_robust = len(df_tsl_rgi['RGI_ID'].unique())
n_glaciers_diff = n_glaciers_orig - n_glaciers_robust
print('Removed '+ str(n_glaciers_diff) +' glaciers with less than '+ str(n_obs_threshold) +' measurements ('+ str(n_glaciers_robust) +' remaining).')
'''

df_tsl_rgi['GTN_code'] = df_tsl_rgi.O1Region.astype(str) + '-0' + df_tsl_rgi.O2Region.astype(str)


# Import GTN-G glacier region tabele
df_gtng_regions = pd.read_csv(input_path_rgi + region_file)
df_gtng_regions.rename(columns={'RGI_CODE': 'GTN_code', 'FULL_NAME': 'region_name'}, inplace=True)


df_tsl_rgi = pd.merge(df_tsl_rgi, df_gtng_regions, how='left', on='GTN_code')


glacier_ids = df_tsl_rgi.RGI_ID.unique()

# glacier_id = 'RGI60-13.04890'
# glacier_ids = glacier_ids[5000:5005,]
df_glacier_maxima = pd.DataFrame(columns=['Year', 'RGI_ID', 'Max_TSL', 'Max_TSL_norm', 'region_name'])
n_runs = 0
n_glaciers_limited = len(glacier_ids)

for glacier_id in glacier_ids:
    progress = (n_runs + 1) / n_glaciers_limited * 100
    print('\n\nWorking on ' + str(glacier_id) + ' ['+ str(n_runs) +' of '+ str(n_glaciers_limited) +' - '+ str(round(progress, 4)) +' %] ...')
    glacier_bool = df_tsl_rgi['RGI_ID'] == glacier_id
    glacier = df_tsl_rgi.loc[glacier_bool]
    glacier.index = pd.to_datetime(glacier['LS_DATE'])
    glacier_annual_max = glacier.loc[glacier.groupby(glacier.index.year)["SC_median"].idxmax()]
    # glacier_annual_max = glacier['SC_median'].resample("Y").max()
    total_max_tsl = glacier_annual_max['SC_median'].max()
    glacier_total_max_bool = glacier_annual_max.SC_median == total_max_tsl
    glacier_total_max_date = glacier_annual_max[glacier_total_max_bool]
    print('Maximum of '+ str(int(glacier_total_max_date.SC_median[0])) +' in '+ str(glacier_total_max_date.index.year[0]))
    df_glacier_maxima = df_glacier_maxima.append({
        'Year': glacier_total_max_date.index.year[0], 
        'Month': cal.month_abbr[glacier_total_max_date.index.month[0]], 
        'RGI_ID': glacier_id, 
        'Max_TSL': int(glacier_total_max_date.SC_median[0]),
        'Max_TSL_norm': glacier_total_max_date.TSL_normalized[0], 
        'region_name': glacier_total_max_date.region_name[0]}, ignore_index=True)
    n_runs += 1

df_glacier_maxima.head()   

df_glacier_maxima_grouped = df_glacier_maxima.groupby(['Month','Year']).agg(['count'])

df_pivot = df_glacier_maxima_grouped.reset_index()
df_pivot['tsl_max_count'] = df_pivot.RGI_ID['count']
df_pivot = df_pivot.pivot(index='Year', columns='Month', values='tsl_max_count')


# From http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
#colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
colors = ['#1f78b4','#b2df8a','#33a02c','#cab2d6','#6a3d9a','#fdbf6f','#ff7f00','#fb9a99','#e31a1c', '#be8876','#b15928', '#a6cee3']

ax = df_pivot.loc[:,['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']].plot.bar(stacked=True, color=colors, figsize=(16,9))
#ax = df_pivot.loc[:,['1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0','11.0','12.0']].plot.bar(stacked=True, colormap='tab20b', figsize=(16,9))
"""
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
"""
print('\nProcessing finished.\n')
