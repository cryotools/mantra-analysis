#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot full TSLA dataset into one time series figure.

@author: David Loibl
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
# import matplotlib.cm as cm
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import sys, os
sys.path.append(os.path.abspath('<PATH_TO_SCM>'))
import ScientificColourMaps6 as SCM6

output_path_plots   = '<OUTPUT_PATH_PLOTS>'
output_file_name    = 'MANTRA-timeseries'
input_path_rgi      = './data/RGI/'

input_file          = '<MANTRA_RESULT_FILE>.h5' 


rgi_filename        = 'rgi60_Asia.csv' 

plot_title          = 'Transient snowline altitudes derived from Landsat'
x_axis_label        = 'Year CE'
y_axis_label        = 'Transient snowline altitude [m a.s.l]'
# 'Transient snowline altitudes derived from Landsat [filtered by intrinsic characteristics and for accumulation phase maxima] '
# plot_title = 'Transient snowlines derived from Landsat [beware: raw, unfiltered data!] '

timeseries_cmap     = SCM6.batlow.reversed() # 'nipy_spectral_r' #'cubehelix_r' # SCM6.batlow.reversed()
dpi                 = 300       # Plot resolution

print('\nReading input file. This may take a while ...')
# HDF Import

df_tsl = pd.read_hdf(input_file, header=0, usecols=['LS_DATE','SC_median', 'TSL_normalized'], parse_dates=['LS_DATE'])
df_tsl.index = pd.to_datetime(df_tsl['LS_DATE'])

# Import RGI data
df_rgi = pd.read_csv(input_path_rgi + rgi_filename, index_col='RGIId', parse_dates=True, low_memory=False)
# df_rgi_sorted = df_rgi.sort_values(by='Area', ascending=False)
df_rgi['RGI_ID'] = df_rgi.index
df_rgi.head()

# Import
print('\nReading TSL input file. This may take a while ...')
df_tsl = pd.read_hdf(input_path_tslprep + input_file, parse_dates=True, index_col='LS_DATE', low_memory=False)
print('Success. Data frame contains '+ str(df_tsl.shape[0]) +' rows and '+ str(df_tsl.shape[1]) +' columns. \n')
#df_tsl.head()

n_obs_orig = df_tsl.shape[0]
n_glaciers_orig = len(df_tsl['RGI_ID'].unique())

# Merge dataframe with RGI data
df_tsl_rgi = pd.merge(df_tsl, df_rgi, how='left', on='RGI_ID')
df_tsl_rgi.index = pd.to_datetime(df_tsl_rgi['LS_DATE'])

n_obs_per_year = df_tsl.groupby(df_tsl.index.year).count()
n_obs_per_year['date'] = pd.to_datetime(n_obs_per_year.index, format='%Y')
n_obs_per_year.set_index('date', inplace=True)
n_obs_per_year.rename(columns={n_obs_per_year.columns[0]: 'n_obs'}, inplace=True)

drop_cols = [x for x in n_obs_per_year.columns if x != 'n_obs']
n_obs_per_year.drop(columns=drop_cols, inplace=True)
n_obs_per_year.plot.bar()


# Plot the time series
print('\nPreparing the time series plot ...')

plt.clf()

plt.rc('font', size=14) #controls default text size
plt.rc('axes', titlesize=14) #fontsize of the title
plt.rc('axes', labelsize=18) #fontsize of the x and y labels
plt.rc('xtick', labelsize=14) #fontsize of the x tick labels
plt.rc('ytick', labelsize=14) #fontsize of the y tick labels
plt.rc('legend', fontsize=14) #fontsize of the legend

fig, ax  = plt.subplots(1,1)
fig.set_size_inches(16,9)


# Preare date formatting
years = mdates.YearLocator(5, month=1, day=1)   # every year
months = mdates.MonthLocator(1)  # every month
years_fmt = mdates.DateFormatter('%Y')

# format the x-axis ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# y-axis limits
ax.set_ylim([2500,6500])

# format the y-axis ticks
#start, end = ax.get_ylim()
#ax.yaxis.set_ticks(np.arange(start, end, 200))
ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(200))


# round to nearest years.
datemin = np.datetime64(df_tsl_rgi['LS_DATE'][0], 'Y') - np.timedelta64(1, 'Y')
datemax = np.datetime64(df_tsl_rgi['LS_DATE'][-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

ax.grid(False)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
#fig.autofmt_xdate()


# Set color mappable
range_min = df_tsl_rgi.CenLat.min()
range_max = df_tsl_rgi.CenLat.max()
cmap = plt.cm.ScalarMappable(
      norm = mcolors.Normalize(range_min, range_max), 
      cmap = plt.get_cmap(timeseries_cmap))

#for i in polygonDict.keys():
#    ax.add_patch(ds.PolygonPatch(polygonDict[i], fc = cmap.to_rgba(df.col1.loc[i])))

cmap.set_array([]) # or alternatively cmap._A = []

ax.scatter(
    x=df_tsl_rgi.LS_DATE, 
    y=df_tsl_rgi.SC_median, 
    marker='.', 
    edgecolors='none', 
    facecolors='full', 
    c=df_tsl_rgi.CenLat, 
    cmap=timeseries_cmap, 
    s=0.05, 
    alpha=1) #c=df_tsl_rgi.CenLat

ax.set_title(plot_title)
ax.set_xlabel(x_axis_label)
ax.set_ylabel(y_axis_label)
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='solid', linewidth=0.1)
ax.yaxis.grid(color='gray', linestyle='none', linewidth=0.1)

#ax.xaxis.set_ticks(np.arange(1985, 2020, 5))
# ax.set_xlim([1984,2000])



n_obs_str = '{:,}'.format(df_tsl_rgi.shape[0])
ax.text(0.01, 0.99, 'n = '+ n_obs_str, ha='left', va='top', transform=ax.transAxes)

# fig.colorbar(cmap, ax = ax)
cbar = fig.colorbar(cmap, ax=ax, pad=0.01)
# cbar.ax.set_yticklabels(['0','1','2','>3'])
cbar.set_label('Latitude [°N]', rotation=270)

# plt.show()

plt.savefig(output_path_plots + output_file_name +'_'+ str(dpi) +'dpi.png', dpi=dpi)

print('\nProcessing finished')
