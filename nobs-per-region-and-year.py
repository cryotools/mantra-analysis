#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:38:18 2020

Create a plot representing a grid of circles for each year and region. 
The number of obsversation for each year and region is shown by circle size.

Inspired by the code of 'ImportanceOfBeingErnest' at StackOverflow:
https://stackoverflow.com/questions/42721302/python-plot-multiple-circle-on-grid-with-legend

@author: David Loibl
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import pandas as pd
import numpy as np


# Configuration
n_obs_threshold     = 100
dpi                 = 300

# Path setup
input_path_rgi     = './data/RGI/'
input_path_tslprep = './data/MANTRA/preprocessed/'
output_path_plots  = '<OUTPUT_PATH_PLOTS>'
output_path_tables = '<OUTPUT_PATH_PLOTS>'

# File setup
input_file         = '<YOUR_MANTRA_FILE>.h5'
rgi_file           = 'rgi60_Asia.csv'
region_file        = 'GTN-G-glacier-subregions.csv'


def wedge_circle(center, radius, angle=0, ax=None, colors=('w','k'), **kwargs):
    """
    Add two half circles to the axes *ax* (or the current axes) with the 
    specified facecolors *colors* rotated at *angle* (in degrees).
    """    
    if ax is None:
        ax = plt.gca()
    '''
    theta1, theta2 = angle, 360 - angle
    w1 = Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    w2 = Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
    '''
    w1 = Wedge(center, radius, 0, angle, fc=colors[0], **kwargs)
    w2 = Wedge(center, radius, angle, 360, fc=colors[1], **kwargs)
    for wedge in [w1, w2]:
        ax.add_artist(wedge)
    return [w1, w2]


# Import TSLA data
print('\nReading TSL file. This may take a while ...')
df_tsl = pd.read_hdf(input_path_tslprep + input_file, parse_dates=True, columns=['RGI_ID', 'SC_median', 'LS_DATE'], index_col='LS_DATE', low_memory=False)
print('Success. Data frame contains '+ str(df_tsl.shape[0]) +' rows and '+ str(df_tsl.shape[1]) +' columns. \n')

n_obs_orig = df_tsl.shape[0]
n_glaciers_orig = len(df_tsl['RGI_ID'].unique())

df_tsl.index = pd.to_datetime(df_tsl['LS_DATE'])

# df_tsl['n_obs_per_glacier'] = df_tsl.groupby(['RGI_ID'])['RGI_ID'].transform('count')

# df_n_obs = df_tsl.groupby('RGI_ID').first()
# df_n_obs.reset_index(inplace=True)


# Import RGI data
df_rgi = pd.read_csv(input_path_rgi + rgi_file, index_col='RGIId', parse_dates=True, low_memory=False)
# df_rgi_sorted = df_rgi.sort_values(by='Area', ascending=False)
df_rgi = df_rgi[df_rgi['Area'] > 0.5]
df_rgi['RGI_ID'] = df_rgi.index
df_rgi.rename(columns={'FiTRegionCode': 'GTN_code', 'FiTRegionLabel': 'region_name'}, inplace=True)

# Create minimal RGI dataframe
rgi_cols = list(df_rgi.columns)
del_cols = [x for x in rgi_cols if x not in ['GTN_code', 'region_name', 'RGI_ID']]
df_rgi_mini = df_rgi.drop(columns=del_cols)



# Merge dataframe with RGI data
df_tsl_rgi = pd.merge(df_tsl, df_rgi_mini, how='left', on='RGI_ID')

df_tsl_rgi.index = pd.to_datetime(df_tsl['LS_DATE'])
# df_tsl_rgi.set_index('LS_DATE', inplace=True)

unique_regions = df_rgi_mini['GTN_code'].unique()
unique_regions.sort()

df_tsl_rgi.sort_values('GTN_code', inplace=True)


years_unique = df_tsl.index.year.unique().to_list()
years_unique.sort()

regions_unique = list(df_rgi_mini['region_name'].unique())
regions_unique.sort()

df_ry_grid = pd.DataFrame(columns=['region_name', 'year', 'n_obs', 'n_obs_jaso', 'n_obs_jaso_prec'])

fig, ax = plt.subplots(figsize=[len(years_unique), len(regions_unique)])

i = 0

for year in years_unique:
    j = 0
    print('\n- - - - '+ str(year) +' - - - -\n')    
    df_subset = df_tsl_rgi[(df_tsl_rgi.index.year == year)]
    df_subset.sort_values(by=['region_name'])
    for region in regions_unique:
        # 
        n_obs = df_subset[(df_subset['region_name'] == region)]['RGI_ID'].count()
        if n_obs > 0:
            n_obs_jaso = df_subset[(df_subset['region_name'] == region) & (df_subset.index.month >= 7) & (df_subset.index.month <= 10)]['RGI_ID'].count()
            n_obs_jaso_perc = n_obs_jaso / n_obs * 100
            wedge_circle((i, -1*j), radius=0.003 * np.sqrt(n_obs), colors=('palevioletred', 'royalblue'), angle=n_obs_jaso_perc * 3.6, ax=ax) # 0.07 * np.log(n_obs)
            ax.axis('equal')
        else:
            n_obs_jaso = 0
            n_obs_jaso_perc = 0
        
        print(str(n_obs) +' ('+ str(n_obs_jaso) +' JASO) obs for '+ str(region) +' in '+ str(year)  )
        df_ry_grid = df_ry_grid.append({'region_name': region, 'year': year, 'n_obs': n_obs, 'n_obs_jaso': n_obs_jaso, 'n_obs_jaso_prec': n_obs_jaso_perc}, ignore_index=True)
        #['RGI_ID'].transform('count')
        j += 1
    i += 1


for ii in range(0, len(years_unique)):
    plt.annotate(xy=(ii, 1), text = str(years_unique[ii]), fontsize = 20, verticalalignment='center', horizontalalignment='center')

for jj in range(0, len(regions_unique)):
    plt.annotate(xy=(-1, -1*jj), text = str(regions_unique[jj]), fontsize = 20, verticalalignment='center', horizontalalignment='right')
    
# ax.set_xlim(-1, len(regions_unique))
# ax.set_ylim(-1 * len(years_unique),3)
plt.axis("off")
plt.plot([-1, len(years_unique)], [-1 * len(regions_unique), 3], alpha=0)
plt.tight_layout()
fig.savefig(output_path_plots, dpi=dpi)

print('\nProcessing finished.\n')

'''
def dual_half_circle(center, radius, angle=0, ax=None, colors=('w','k'), **kwargs):
    """
    Add two half circles to the axes *ax* (or the current axes) with the 
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    w1 = Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    w2 = Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
    for wedge in [w1, w2]:
        ax.add_artist(wedge)
    return [w1, w2]

DF = pd.DataFrame(np.random.choice([True, False], size = (15, 10)))

fig, ax = plt.subplots(figsize=(8,13))
for ii in range(0,DF.shape[1]):    
    for jj in range(0,DF.shape[0]):
        if DF[ii][jj]:
            dual_half_circle((ii, -1*jj), radius=0.3, colors=('b','g'), angle=90, ax=ax)
            ax.axis('equal')

for ii in range(0,DF.shape[1]):       
    plt.annotate(xy= (ii, 1), s= 'W'+str(ii), fontsize = 10, verticalalignment='center', horizontalalignment='center')

for jj in range(0,DF.shape[0]):
    plt.annotate(xy =(-1, -1*jj),s= 'subj '+str(jj), fontsize =10, verticalalignment='center', horizontalalignment='right')

ax.set_xlim(-1,10)
ax.set_ylim(-15,3)
plt.axis("off")
plt.plot([-1,10], [-15,3], alpha=0)
plt.tight_layout()
plt.show()
'''
