#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Create a stacked bar plot displaying the number of glaciers that have their
maxima in each year.

@author: David Loibl
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

#import sys
#import os
#import re


# Path setup
input_path_rgi     = './data/RGI/'
input_path_tsl = './data/MANTRA/'
path_plots  = '<PATH_TO_STORE_PLOTS>'
path_tables = '<PATH_TO_STORE_TABLES>'

Path(path_plots).mkdir(parents=True, exist_ok=True)
Path(path_tables).mkdir(parents=True, exist_ok=True)


# Configuration
use_fit_regions    = True
rgi_filename       = 'rgi60_Asia.csv'
input_file         = '<MANTRA_RESULTS>.h5'
region_file        = '<GTNG_REGIONS>.csv'

use_preproc_data   = False      # True -> Read data from existing files
                                # False -> Preprocess with this run

n_obs_threshold    = 100        # Set to 0 to deactivate n obs threshold filtering
area_threshold     = 2          # Set to 0 to deactivate area threshold filtering

tsl_full_file      = 'TSL-maxima.csv'
tsl_area_file      = 'TSL-maxima-area_thres_'+ str(area_threshold) +'.csv'
tsl_nobs_file      = 'TSL-maxima-n_obs_thres_'+ str(n_obs_threshold) +'.csv'



def prepare_TSL_data():
    # Import RGI data
    df_rgi = pd.read_csv(input_path_rgi + rgi_filename, index_col='RGIId', parse_dates=True, low_memory=False)
    # df_rgi_sorted = df_rgi.sort_values(by='Area', ascending=False)
    df_rgi['RGI_ID'] = df_rgi.index
    df_rgi.head()
    
    # Import
    print('\nReading TSL input file. This may take a while ...')
    df_tsl = pd.read_hdf(input_path_tsl + input_file, parse_dates=True, index_col='LS_DATE', low_memory=False)
    print('Success. Data frame contains '+ str(df_tsl.shape[0]) +' rows and '+ str(df_tsl.shape[1]) +' columns. \n')
    #df_tsl.head()
    
    n_obs_orig = df_tsl.shape[0]
    n_glaciers_orig = len(df_tsl['RGI_ID'].unique())
    
    # Merge dataframe with RGI data
    df_tsl_rgi = pd.merge(df_tsl, df_rgi, how='left', on='RGI_ID')
    df_tsl_rgi.index = pd.to_datetime(df_tsl_rgi['LS_DATE'])
    
    df_tsl_rgi['n_obs'] = df_tsl_rgi.groupby(['RGI_ID'])['RGI_ID'].transform('count')
        
    
    print('Extracting ablation phase subset.')
    ablphase_subset = np.where((df_tsl_rgi.index.month >= 8) & (df_tsl_rgi.index.month <= 10))[0]
    df_tsl_rgi = df_tsl_rgi.iloc[ablphase_subset,:]
    
    df_tsl_rgi['n_obs_abl'] = df_tsl_rgi.groupby(['RGI_ID'])['RGI_ID'].transform('count')
    
    n_obs_abl = df_tsl_rgi.shape[0]
    n_obs_diff = n_obs_orig - n_obs_abl
    print('Removed '+ str(n_obs_diff) +' accumulation phase observations ('+ str(n_obs_abl) +' remaining).')
    
    
    
    return df_tsl_rgi



def TSL_annual_maxima(df_tsl_rgi, export_file=None):
    glacier_ids = df_tsl_rgi.RGI_ID.unique()
    
    # glacier_id = 'RGI60-13.04890'
    # glacier_ids = glacier_ids[5000:5005,]
    df_glacier_maxima = pd.DataFrame(columns=['Year', 'RGI_ID', 'Max_TSL', 'Max_TSL_norm', 'region_name', 'region_code'])
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
            'Date': glacier_total_max_date['LS_DATE'][0],
            'LS_ID': glacier_total_max_date['LS_ID'][0],
            'RGI_ID': glacier_id, 
            'Max_TSL': int(glacier_total_max_date.SC_median[0]),
            'Max_TSL_norm': glacier_total_max_date.TSL_normalized[0], 
            'n_obs_abl': glacier_total_max_date['n_obs_abl'][0],
            'n_obs': glacier_total_max_date['n_obs'][0],
            'glacier_area': glacier_total_max_date['glacier_area'][0],
            'region_name': glacier_total_max_date.region_name[0],
            'region_code': glacier_total_max_date['GTN_code'][0]
            }, ignore_index=True)
        n_runs += 1

    if export_file != None:
        df_glacier_maxima_export = df_glacier_maxima.copy()
        # df_glacier_maxima_export.drop(columns=['Year'], inplace=True) 
        df_glacier_maxima_export['Max_TSL'] = df_glacier_maxima['Max_TSL'].astype('float')
        # df_glacier_maxima_export['Date'] = df_glacier_maxima['Date'].astype('datetime')
        # df_glacier_maxima_export.to_hdf(output_path_tables + 'TSL-maxima.h5', key='maxima', mode='a', format='table', data_columns=['RGI_ID'])
        df_glacier_maxima_export.to_csv(path_tables + export_file)            
        
    return df_glacier_maxima


def make_pivot(df_glacier_maxima):
    if 'Year' not in df_glacier_maxima.columns:
        df_glacier_maxima['Year'] = pd.DatetimeIndex(df_glacier_maxima['Date']).year
    
    df_glacier_maxima_grouped = df_glacier_maxima.groupby(['region_name','Year']).agg(['count'])

    df_pivot = df_glacier_maxima_grouped.reset_index()
    df_pivot['tsl_max_count'] = df_pivot.RGI_ID['count']
    df_pivot = df_pivot.pivot(index='Year', columns='region_name', values='tsl_max_count')
    
    '''
    if use_fit_regions:
        mountain_ranges = list(df_regions['region_label'])
    else: 
        mountain_ranges = list(df_pivot.columns)
    '''
    mountain_ranges = list(df_glacier_maxima['region_name'].unique())
    df_plot = df_pivot.loc[:,mountain_ranges]
    
    return df_plot


def make_stacked_barplot(conf_dict):
    
    sns.set(rc=conf_dict['rc'])
    # sns.set_context("paper")
    sns.set_style("ticks", {
        'xtick.bottom': True, 
        'axes.facecolor': '.95', 
        'axes.edgecolor': '.8',
        'axes.grid': True,
        'grid.color': '.9',
        'grid.linestyle': '-',
        'patch.edgecolor': 'g',
        'patch.force_edgecolor': False,
        })
    
    from matplotlib.colors import ListedColormap
      
    palette_cmap = ListedColormap(conf_dict['palette'].as_hex())

    ax = conf_dict['pivot_table'].plot(kind='bar', stacked=True, linewidth=0, colormap=palette_cmap)
    # ax = plt.subplot(111,aspect = 'equal')
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.99, top=0.95, wspace=0.05, hspace=0)
    # ax = sns.boxplot(x=conf_dict['x'], y=conf_dict['y'], palette=conf_dict['palette'])
    # ax = df_n_obs.boxplot(column='n_obs_per_glacier', by='region_name', figsize=(20,9))
    ax.set_title(conf_dict['title'], fontsize=16)
    ax.set_xlabel(conf_dict['x_label'], fontsize=14)
    ax.set_ylabel(conf_dict['y_label'], fontsize=14)    
    ax.set(xlim=conf_dict['x_lim'])
    ax.set(ylim=conf_dict['y_lim'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  
    ax.text(  # position text relative to Figure            
        0.01, 0.99, conf_dict['text'],
        ha='left', va='top', transform=ax.transAxes
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=8)

    # plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(conf_dict['output_file'])
    plt.close()






def subset_by_threshold(df, column_name, threshold, greater=False):    
    if greater:
        operator = '<'        
        subset = np.where(df[column_name] <= threshold)[0]
        
    else:
        operator = '>'          
        subset = np.where(df[column_name] >= threshold)[0]
    
    print('Extracting subset with '+ str(column_name) +' '+ operator +' '+ str(threshold) +' ...')
    
    df = df.iloc[subset,:]
    n_glaciers_robust = len(df['RGI_ID'].unique())
    n_glaciers_orig = df.shape[0]
    n_glaciers_diff = n_glaciers_orig - n_glaciers_robust
    print('Removed '+ str(n_glaciers_diff) +' glaciers  with '+ str(column_name) +' '+ operator +' '+ str(area_threshold) +'  ('+ str(n_glaciers_robust) +' remaining).')
    
    return df
    




# Read TSL data
if use_preproc_data is False:
    df_tsl_rgi = prepare_TSL_data()
    
    # Handle regions
    if use_fit_regions:
        df_tsl_rgi.rename(columns={'FiTRegionCode': 'GTN_code', 'FiTRegionLabel': 'region_name'}, inplace=True)
        df_regions = pd.read_csv(input_path_rgi + region_file)
    else:    
        df_tsl_rgi['GTN_code'] = df_tsl_rgi.O1Region.astype(str) + '-0' + df_tsl_rgi.O2Region.astype(str)
        
        # Import GTN-G glacier region tabele
        df_gtng_regions = pd.read_csv(input_path_rgi +'GTN-G-glacier-subregions.csv')
        df_gtng_regions.rename(columns={'RGI_CODE': 'GTN_code', 'FULL_NAME': 'region_name'}, inplace=True)
        
        df_tsl_rgi = pd.merge(df_tsl_rgi, df_gtng_regions, how='left', on='GTN_code')
                
    df_glacier_maxima_full = TSL_annual_maxima(df_tsl_rgi, tsl_full_file)
    
    if area_threshold > 0:
        df_tsl_rgi_area_thres = subset_by_threshold(df_tsl_rgi, 'glacier_area', area_threshold)
        df_glacier_maxima_area = TSL_annual_maxima(df_tsl_rgi_area_thres, tsl_area_file)
    
    if n_obs_threshold > 0:
        df_tsl_rgi['n_obs_per_glacier'] = df_tsl_rgi.groupby(['RGI_ID'])['RGI_ID'].transform('count')
        df_tsl_rgi_nobs_thres = subset_by_threshold(df_tsl_rgi, 'n_obs_per_glacier', n_obs_threshold)   
        df_glacier_maxima_nobs = TSL_annual_maxima(df_tsl_rgi_nobs_thres, tsl_nobs_file)
                
else:
    df_glacier_maxima_full = pd.read_csv(path_tables + tsl_full_file)    
    df_plot_full = make_pivot(df_glacier_maxima_full)
        
    if area_threshold > 0:
        df_glacier_maxima_area = pd.read_csv(path_tables + tsl_area_file)
        
    if n_obs_threshold > 0:        
        df_glacier_maxima_nobs = pd.read_csv(path_tables + tsl_nobs_file)       




# Make pivot tables and plots
print('\nPreparing bar plot for full TSL dataset')
df_glacier_maxima_full.sort_values('region_code', inplace=True)
df_plot_full = make_pivot(df_glacier_maxima_full)
conf_dict_full = {
    'pivot_table': df_plot_full,
    'palette': sns.hls_palette(len(df_glacier_maxima_full['region_name'].unique()), h=.5),
    # 'palette': sns.color_palette("hls", df_regions.shape[0], reverse=True),
    'title': 'Year of maximum TSL elevation at individual glaciers throughout the respective time series',
    'x_label': 'Year CE',
    'x_lim': (None, None),
    'y_label': 'Number of glaciers',
    'y_lim': (-20, None),
    'text': 'n glaciers  = '+ str(len(df_glacier_maxima_full)),
    'rc': {'figure.figsize':(20,12)},
    'output_file': path_plots + 'hist-total-TSL-max-year-by-region.png'
}

make_stacked_barplot(conf_dict_full)


if area_threshold > 0:
    print('\nPreparing bar plot for TSL subset with glacier area > '+ str(area_threshold))
    df_glacier_maxima_area.sort_values('region_code', inplace=True)
    df_plot_area = make_pivot(df_glacier_maxima_area)
    
    conf_dict_area = {
        'pivot_table': df_plot_area,
        'palette': sns.hls_palette(len(df_glacier_maxima_area['region_name'].unique()), h=.5),
        # 'palette': sns.color_palette("hls", df_regions.shape[0], reverse=True),
        'title': 'Year of maximum TSL elevation at glaciers throughout the respective time series, area threshold '+ str(area_threshold) +' km²',
        'x_label': 'Year CE',
        'x_lim': (None, None),
        'y_label': 'Number of glaciers',
        'y_lim': (-20, None),
        'text': 'n glaciers  = '+ str(len(df_glacier_maxima_area)), #  +'\nn measurements = '+ str(n_obs_orig)
        'rc': {'figure.figsize':(20,12)},
        'output_file': path_plots + 'hist-total-TSL-max-year-by-region-glaciers-gte-'+ str(area_threshold) +'sqkm.png'
    }
    
    make_stacked_barplot(conf_dict_area)

if n_obs_threshold > 0:
    print('\nPreparing bar plot for TSL subset with n obs > '+ str(n_obs_threshold))
    df_glacier_maxima_nobs.sort_values('region_code', inplace=True)
    df_plot_nobs = make_pivot(df_glacier_maxima_nobs)

    conf_dict_nobs = {
        'pivot_table': df_plot_nobs,
        'palette': sns.hls_palette(len(df_glacier_maxima_nobs['region_name'].unique()), h=.5),
        # 'palette': sns.color_palette("hls", df_regions.shape[0], reverse=True),
        'title': 'Year of maximum TSL elevation at glaciers throughout the respective time series, n obs. threshold '+ str(n_obs_threshold),
        'x_label': 'Year CE',
        'x_lim': (None, None),
        'y_label': 'Number of glaciers',
        'y_lim': (-5, None),
        'text': 'n glaciers  = '+ str(len(df_glacier_maxima_nobs)) ,
        'rc': {'figure.figsize':(20,12)},
        'output_file': path_plots + 'hist-total-TSL-max-year-by-region-n_obs-gte-'+ str(n_obs_threshold) +'.png'
    }
    
    make_stacked_barplot(conf_dict_nobs)




'''
ax = df_pivot.loc[:,mountain_ranges].plot.bar(stacked=True, colormap='tab20b', figsize=(16,9)) 

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
'''


print('\nProcessing finished.\n')



