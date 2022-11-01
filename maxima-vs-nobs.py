#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:06:43 2020

@author: David Loibl    
"""

import pandas as pd
import numpy as np
from pathlib import Path
import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy import stats

import sys, os
sys.path.append(os.path.abspath('<SCM_PATH>'))
import ScientificColourMaps6 as SCM6


# Start the clock
start_time = datetime.datetime.now().timestamp()

### SETUP VARS
use_preprc_grid = False 
verbosity       = 1         # Reporting level

norm_SLA_thres  = 0.3        # Do not consider SLA values < threshold, set to 0 to deactivate
min_nobs_thres  = 1         # Exclude data with <= n obs per glacier and yr, set to 0 to consider all data

abl_phase_begin = 7         # Month in which the ablation phase begins
abl_phase_end   = 10        # Month in which the abl. ph. ends (included)

start_year      = 0         # Year to start the analysis. 0 to consider all years
end_year        = 0          # Year to stop the analysis. 0 to consider all years

sla_min_value   = 0.3       # Minimum TSL to be considered a possible SLA

dpi             = 300


# Path setup
input_path_rgi     = './data/RGI/'
input_path_tslprep = './data/MANTRA/preprocessed/'

input_path_pp_grd  = './data/MANTRA/preprocessed/'
input_path_tslaggr = './data/MANTRA/aggregated/'


output_path_plots  = '<OUTPUT_PATH_PLOTS>'
output_path_tables = '<OUTPUT_PATH_PLOTS>'

Path(output_path_plots).mkdir(parents=True, exist_ok=True)
Path(output_path_tables).mkdir(parents=True, exist_ok=True)

# File setup
input_file_tsl     = '<YOUR_MANTRA_FILE>.h5'
rgi_file           = 'rgi60_Asia.csv'
pp_grid_file       = '<GRID_FILE>.h5' 


### FUNCTIONS

def make_subset_by_months(df, min_month, max_month):
    subset_rule = np.where((df.index.month >= min_month) & (df.index.month <= max_month))[0]
    subset = df.iloc[subset_rule,:]
    
    return subset.copy()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



### IMPORT DATA

# Import TSLA data
print('\nReading TSLA file. This may take a while ...')
df_tsl = pd.read_hdf(input_path_tslprep + input_file_tsl, parse_dates=True, columns=['RGI_ID', 'SC_median', 'TSL_normalized', 'LS_DATE'], index_col='LS_DATE', low_memory=False) 
print('Success. Data frame contains '+ str(df_tsl.shape[0]) +' rows and '+ str(df_tsl.shape[1]) +' columns. \n')   
           
if 'LS_DATE' not in df_tsl.columns:
    df_tsl['LS_DATE'] = df_tsl.index.to_series().dt.strftime('%Y-%m-%d')

df_tsl['date'] = df_tsl['LS_DATE']
df_tsl.index = pd.to_datetime(df_tsl['LS_DATE'])   

if norm_SLA_thres > 0:
    SLA_thres_query = np.where((df_tsl['TSL_normalized'] >= norm_SLA_thres))[0]    
    print('Keeping '+ str(len(SLA_thres_query)) +' observations > SLA threshold ('+ str(norm_SLA_thres) +')')
    df_tsl = df_tsl.iloc[SLA_thres_query, :]

    
n_obs_orig = df_tsl.shape[0]
n_glaciers_orig = len(df_tsl['RGI_ID'].unique())


# Import RGI data
print('\nMerging RGI and TSLA data ...')
df_rgi = pd.read_csv(input_path_rgi + rgi_file, index_col='RGIId', parse_dates=True, low_memory=False)
# df_rgi_sorted = df_rgi.sort_values(by='Area', ascending=False)
df_rgi['RGI_ID'] = df_rgi.index
df_rgi = df_rgi[df_rgi['Area'] > 0.5]
df_rgi.rename(columns={'FiTRegionCode': 'region_code', 'FiTRegionLabel': 'region_name', 'CenLon': 'longitude', 'CenLat': 'latitude'}, inplace=True)

# Create minimal RGI dataframe
rgi_cols = list(df_rgi.columns)
del_cols = [x for x in rgi_cols if x not in ['region_code', 'region_name', 'RGI_ID', 'longitude', 'latitude']]
df_rgi_mini = df_rgi.drop(columns=del_cols)


# Merge dataframe with RGI data
df_tsl = pd.merge(df_tsl, df_rgi_mini, how='left', on='RGI_ID')
df_tsl.index = pd.to_datetime(df_tsl['LS_DATE'])

# Prepare seasonal subsets
print('\nExtracting ablation phase subset for values > '+ str(sla_min_value) +' norm. TSLA ...')
abl_phase_query = np.where((df_tsl.index.month >= abl_phase_begin) & 
                           (df_tsl.index.month <= abl_phase_end) &
                           (df_tsl['TSL_normalized'] > sla_min_value))[0]
    
df_tsl_summer = df_tsl.iloc[abl_phase_query,:].copy()
df_tsl_summer['year'] = df_tsl_summer.index.year    

print('Ablation phase subset contains '+ str(df_tsl_summer.shape[0]) +' rows.\n')   

# Add n_obs_column
df_tsl_summer.value_counts(['RGI_ID', 'year']).reset_index(name='n_obs')

# Optionally, drop rows where n obs per glacier per year are below threshold
if min_nobs_thres > 0:
    df_tsl_summer = df_tsl_summer[df_tsl_summer['n_obs'] > min_nobs_thres ]



# Prepare list of unique years; drop years if neccessary.
years_unique = df_tsl.index.year.unique().to_list()
years_unique.sort()

if start_year == 0:        
    start_year = years_unique[0]
else:
    years_unique = [ item for item in years_unique if item >= start_year ]    

if end_year == 0:
    end_year = years_unique[-1]
else:
    years_unique = [ item for item in years_unique if item <= end_year ]    
    

glaciers_unique = list(df_rgi['RGI_ID'].unique())

 

n_obs_per_year = df_tsl_summer.groupby(['RGI_ID', 'year'])['n_obs'].count()
TSLnd_annl_max = df_tsl_summer.groupby(['RGI_ID', 'year'])['TSL_normalized'].max() 

nobs_vs_maxima_data = {'n_obs': n_obs_per_year, 'TSLA_norm_annual_max': TSLnd_annl_max}

df_nobs_vs_maxima = pd.concat(nobs_vs_maxima_data, axis=1)
df_nobs_vs_maxima.reset_index(inplace=True)

df_nobs_vs_maxima_rgi = pd.merge(df_nobs_vs_maxima, df_rgi_mini, how='left', on='RGI_ID')




# Make spatial grid
lons = df_tsl['longitude'][:].values
lats = df_tsl['latitude'][:].values

lat_max = np.ceil(np.amax(lats))
lat_min = np.floor(np.amin(lats))

lon_max = np.ceil(np.amax(lons))
lon_min = np.floor(np.amin(lons))

grid_lat = np.linspace(lat_min, lat_max, int(lat_max - lat_min + 1))
grid_lon = np.linspace(lon_min, lon_max, int(lon_max - lon_min + 1))

# - - - - - - - PREPARE SUBSETS - - - - - - - - - - 
# Group by:
# 1 Meteor. season: 		        DJF, MAM, JJA, SON or
# 2 adapted to glacial cylcle	JFM, AMJ, JAS, OND or
# 3 winter vs. summer		    NDJFM, JJAS

df_tsl.index = pd.to_datetime(df_tsl['LS_DATE'])



# Prepare list of unique years; drop years if neccessary.

years_unique = df_tsl.index.year.unique().to_list()
years_unique.sort()

if start_year == 0:        
    start_year = years_unique[0]
else:
    years_unique = [ item for item in years_unique if item >= start_year ]    


if end_year == 0:
    end_year = years_unique[-1]
else:
    years_unique = [ item for item in years_unique if item <= end_year ]    
    

# lat, lon = lat_min, lon_min
grid = np.full([len(years_unique), len(grid_lat), len(grid_lon)], np.nan)


# data_column = 'SC_median'
data_column = 'TSLA_norm_annual_max'

if use_preprc_grid:
    df_grid = pd.read_hdf(input_path_pp_grd + pp_grid_file)
else:
    df_grid = pd.DataFrame(columns=['lat','lon','year','TSLA_norm_mean', 'n_obs_JASO_mean'])
    
    year_counter = 0
    
    for year in years_unique:
        print('\n\nProcessing data for year '+ str(year))
        grid_lat_position = 1
        
        for lat in grid_lat:
            lat = int(lat)
            if(grid_lat_position < len(grid_lat)):
                print('  Latitude: '+ str(int(grid_lat[grid_lat_position])))
                
                grid_lon_position = 1
                
                for lon in grid_lon:
                    lon = int(lon)
                    
                    if(grid_lon_position < len(grid_lon)):                            
                        spatial_subset_rule = np.where((
                            df_nobs_vs_maxima_rgi['latitude'] >= grid_lat[grid_lat_position - 1]) 
                                 & (df_nobs_vs_maxima_rgi['latitude'] < grid_lat[grid_lat_position]) 
                                 & (df_nobs_vs_maxima_rgi['longitude'] >= grid_lon[grid_lon_position - 1]) 
                                 & (df_nobs_vs_maxima_rgi['longitude'] < grid_lon[grid_lon_position])
                                 & (df_nobs_vs_maxima_rgi['year'] == year))[0]
                        
                        spatial_subset = df_nobs_vs_maxima_rgi.iloc[spatial_subset_rule,:]
                        if verbosity > 1:
                            print('    '+ str(len(spatial_subset)) +' obs found in grid cell  '+ str(int(grid_lat[grid_lat_position])) +'/'+ str(int(grid_lon[grid_lon_position])) +'.')
                        
                        # spatial_subset.drop(columns=['latitude', 'longitude'], inplace=True)
                        # annual_subset_TSLA_mean = spatial_subset[data_column].groupby(spatial_subset.index.year).mean()  
                                            
                        if len(spatial_subset) != 0:
                            TSLA_mean = spatial_subset[data_column].groupby(spatial_subset['year']).mean()
                            nobs_mean = spatial_subset['n_obs'].groupby(spatial_subset['year']).mean()
                            df_temp = pd.DataFrame.from_dict({'lat': lat,
                                                 'lon': lon,
                                                 'year': year,
                                                 'TSLA_norm_mean': TSLA_mean, 
                                                 'n_obs_JASO_mean': nobs_mean})
                            
                            df_grid = df_grid.append(df_temp, ignore_index=True)
                            # grid[year_counter][grid_lat_position][grid_lon_position] = TSLA_mean
                            # TSLA_median = spatial_subset[data_column].groupby(spatial_subset.index.year).median()
                            # grid[year_counter][grid_lat_position][grid_lon_position] = TSLA_median
                        # lon_grid.append(annual_subset_TSLA_mean.to_list())
                        # grid.append(annual_subset_TSLA_mean)
                        grid_lon_position += 1
                    
        
                # grid.append(lon_grid)
                grid_lat_position += 1
                # df_grid = df_grid.append(df_temp, ignore_index=True)
        year_counter += 1
        
        df_grid.to_hdf(input_path_pp_grd + pp_grid_file, key='year')

    
       
# grid = loop_through_time_and_space(df_tsl, years_unique, grid_lat, grid_lon)    

# df_grid_filtered = df_grid[(df_grid['n_obs_JASO_mean'] != 1) & (df_grid['n_obs_JASO_mean'] != 2)]
df_grid_filtered = df_grid.copy()

# Linear regression for the whole dataset

# x = np.array(df_grid_filtered['year'], dtype='float64')
x = np.array(df_grid_filtered['n_obs_JASO_mean'], dtype='float64')
y = np.array(df_grid_filtered['TSLA_norm_mean'], dtype='float64')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print("slope: %f    intercept: %f" % (slope, intercept))
print("p value: %f" % p_value)
print("R-squared: %f" % r_value**2)



arr = np.linspace(0, 50, 100).reshape((10, 10))
fig, ax = plt.subplots(ncols=2)

cmap = plt.get_cmap(SCM6.tokyo)
tokyo_cut = truncate_colormap(cmap, 0, 0.7)


plt.figure(figsize=(15,10))
plt.scatter(x, y, c=df_grid_filtered['year'], marker='.', label='original data', s=5, cmap=tokyo_cut)
plt.colorbar()
plt.plot(x, intercept + slope*x, 'r', label='fitted line', linewidth=0.4)
plt.ylabel('normalized TSLA')
plt.xlabel('n JASO observations per year')
plt.title('All measurements - R²: '+ str(np.round(r_value**2, decimals=3)))

fig_name = 'LM-nobs-vs-SLA-HMA.png'
# print('Saving figure ' + fig_name)
plt.savefig(output_path_plots + fig_name, bbox_inches='tight')
plt.close()


# Linear regessions for individual 1° cells

glaczierized_cells = df_grid_filtered.groupby(['lat','lon']).size().reset_index().rename(columns={0:'count'})

lats, lons, slopes, intercepts, r_values, r2_values, p_values, std_errs, ns = [np.nan], [np.nan], [slope], [intercept], [r_value], [r_value**2], [p_value], [std_err], [df_grid_filtered.shape[0]]

for cell in glaczierized_cells.iterrows():
    lon = cell[1]['lon']
    lat = cell[1]['lat']
    df_grid_cell = df_grid_filtered[(df_grid_filtered['lon'] == lon) & (df_grid_filtered['lat'] == lat)]
    
    x = np.array(df_grid_cell['n_obs_JASO_mean'], dtype='float64')
    y = np.array(df_grid_cell['TSLA_norm_mean'], dtype='float64')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    lats.append(lat)
    lons.append(lon)
    slopes.append(slope)
    intercepts.append(intercept)
    r_values.append(r_value)
    r2_values.append(r_value**2)
    p_values.append(p_value)
    std_errs.append(std_err)
    ns.append(df_grid_cell.shape[0])
    
    print("slope: %f    intercept: %f" % (slope, intercept))
    print("R-squared: %f" % r_value**2)
    
    '''
    fig = plt.figure(figsize=(3000/plot_dpi, 1500/plot_dpi), dpi=plot_dpi)
    
    # first plot
    ax = fig.add_subplot(111)
        
    
    ax.axvline(x=xc, color='0.8', linestyle=':', linewidth=0.2)
    
    # Set y axis limits proportional to glacier min/max elevation
    y_min = np.around((current_glacier['glacier_DEM_min'][0] - 100) / 100) * 100
    y_max = np.around((current_glacier['glacier_DEM_max'][0] + 100) / 100) * 100        
    ax.set_ylim([y_min, y_max])
    
    # Add horizontal lines for glacier min/max elevations        
    ax.axhline(y=current_glacier['glacier_DEM_min'][0], linestyle='--', linewidth=0.2, color = 'k')
    ax.axhline(y=current_glacier['glacier_DEM_max'][0], linestyle='--', linewidth=0.2, color = 'k')
    
    # grey dots for suspicious values
    # ax.plot(current_glacier.SC_median, '0.7', marker='.', linestyle='')
    # ax.plot(current_glacier.SC_median, '0.7', marker='', linestyle='-', linewidth=0.1)

    # black star for annual maximum
    ax.plot(anmax, 'k*', zorder=0)
            
    # Blue dots + thin line for ok values
    ax.errorbar(current_glacier.index, current_glacier.SC_median, yerr=current_glacier.SC_stdev, fmt='b.-', ecolor='lightgray', elinewidth=2, linewidth=0.1, zorder=5)
    
    # green dots for summer values
    ax.plot(current_glacier.SC_median[jas], 'g.', zorder=10) 
    
    
        
    # label with the glacier name
    ax.text(0.01, 0.95, glacier_id +' (' + str(round(current_glacier.glacier_area[0], 2)) + ' km²)', transform=ax.transAxes)

    #if j2 == 0:
    #    ax.set_title('Transient snowline time series')
    '''
    
    
    # 
    
    plt.figure(figsize=(15,10))
    plt.scatter(x, y, c=df_grid_cell['year'], marker='.', label='original data', s=3, cmap='viridis')
    plt.colorbar()
    
    plt.plot(x, y, '.', label='original data', markersize=3)
    plt.plot(x, intercept + slope*x, 'r', label='fitted line', linewidth=0.6)
    plt.ylabel('normalized TSLA')
    plt.xlabel('n JASO observations per year')
    plt.title('Meas. in cell lat:'+ str(lat) +'-lon:'+ str(lon) +' R²: '+ str(np.round(r_value**2, decimals=3)))
    # plt.legend()
    
    fig_name = 'LM-nobs-vs-SLA-Lat'+str(lat)+'-Lon'+ str(lon) +'.png'
    # print('Saving figure ' + fig_name)
    plt.savefig(output_path_plots + fig_name, bbox_inches='tight')
    plt.close()
    #plt.show()
    

df_regr = pd.DataFrame.from_dict({'lat': lats,
                                  'lon': lons,
                                  'slope': slopes, 
                                  'intercept': intercepts, 
                                  'r_value': r_values, 
                                  'r2_value': r2_values, 
                                  'p_value': p_values,
                                  'std_err': std_errs,
                                  'n': ns})

df_regr.to_csv(output_path_tables +'LM-nobs-vs-SLA-1degGrid.csv', index=False)






import seaborn as sns


dpi                = 300        # Resolution of output plots in dpi


# Path setup
output_path_plots  = '../output/plots/stats/LM-nobs-vs-SLA/boxplots-nobs-per-yr/'

Path(output_path_plots).mkdir(parents=True, exist_ok=True)
#Path(output_path_tables).mkdir(parents=True, exist_ok=True)

# File setup
# input_file_tsl     = 'TSLA-HMA-2020-07-filtered-nWmax_0_2SLAthres-0_3cut.h5'
# input_file_tsl     = 'TSLA-HMA-2020-07-filtered-nWmax_0_2thres_noDub_JASO.h5'
# 'TSLA-HMA-2020-07-filtered-nWmax_0_2SLAthres-0_3cut--annual-maxima-SLAthres0_3.h5'
# input_file_grid    = 'annual-max-means-normalized-TSLA-1deg_grid-3+nobs.npy'
# 'annual-max-means-normalized-TSLA-1deg_grid.npy'
# 'annual-max-means-normalized-TSLA-1deg_grid.npy' 
# 'annual-means-normalized-TSLA-1deg_grid.npy'
# output_file        = '_grid_export_.npy'
# output_file_pc_plot = 'PC_timeseries'
# rgi_file           = 'rgi60_Asia_fitregions.csv'

output_file_name   = 'boxplot-nobs-per-glacier'


# Import data

'''
# Import TSLA data
print('\nReading TSLA file. This may take a while ...')
df_tsl = pd.read_hdf(input_path_tslprep + input_file_tsl, parse_dates=True, columns=['RGI_ID', 'SC_median', 'TSL_normalized', 'LS_DATE'], index_col='LS_DATE', low_memory=False) 
    
    
# Prepare seasonal subsets
print('Extracting ablation phase subset.')
df_tsl = make_subset_by_months(df_tsl, 7, 10)
# winter_subset = make_subset_by_months(df_tsl, 3, 11)

print('Success. Data frame contains '+ str(df_tsl.shape[0]) +' rows and '+ str(df_tsl.shape[1]) +' columns. \n')   




n_obs_orig = df_tsl.shape[0]
n_glaciers_orig = len(df_tsl['RGI_ID'].unique())
'''    


df_tsl['year'] = df_tsl.index.year

tmp_max = df_tsl.groupby(by=['year', 'RGI_ID'])['TSL_normalized'].max()
tmp_nobs = df_tsl.groupby(by=['year', 'RGI_ID'])['region_code'].count()
df_max_vs_nobs = pd.concat([tmp_max, tmp_nobs], axis=1).reset_index()
df_max_vs_nobs.rename(columns={'region_code': 'n_obs'}, inplace=True)
# df_max_vs_nobs.head()

years_unique = df_max_vs_nobs['year'].unique()
plts = []
for year in years_unique:
    print('Processing data for '+ str(year))
    df_subset = df_max_vs_nobs[df_max_vs_nobs['year'] == year].copy()
    plt.clf()
    sns.boxplot(x='n_obs', y='TSL_normalized', data=df_subset).set_title(str(year))
    plt.savefig(output_path_plots + output_file_name +'-'+ str(year), bbox_inches='tight', dpi=dpi)
    
    #plt.show()
    
'''
df_regr_gt10_sgn99['p_value'].mean() 
df_regr_gt10_sgn99['p_value'].std()
df_regr_gt10_sgn99['p_value'].median()
df_regr_gt10_sgn99['p_value'].mad()

df_regr_gt10_sgn99['r2_value'].mean()
df_regr_gt10_sgn99['r2_value'].std()
df_regr_gt10_sgn99['r2_value'].median()
df_regr_gt10_sgn99['r2_value'].mad()
'''



end_time = datetime.datetime.now().timestamp()
proc_time = end_time - start_time


print('\nProcessing finished in '+ str(datetime.timedelta(seconds=proc_time)) +'\n')
