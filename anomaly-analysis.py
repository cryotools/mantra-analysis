#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 07:55:26 2019

@author: David Loibl
"""

import sys, os
sys.path.append(os.path.abspath('./lib'))
from plotting import plot_anomaly_timeseries, plot_grid_map
from gridding import annual_lat_lon_grid

sys.path.append(os.path.abspath('<PATH_TO_SCM>'))
import ScientificColourMaps6 as SCM6

import pandas as pd
import numpy as np
from pathlib import Path

import datetime
start_time = datetime.datetime.now().timestamp()



# Configuration
# n_obs_threshold = 100
start_year         = 0       # Year to start the analysis. 0 to consider all years
end_year           = 0       # Year to stop the analysis. 0 to consider all years

# Conf for plots with temporal bins
bin_start_year     = 1985
bin_size           = 5
n_bins             = 7

# Plot design options
vmin               = -150       # Minimum of value range for color map
vmax               = 150        # Maximum of value range for color map
dpi                = 300        # Resolution of output plots in dpi

# Data configuration
run_identifier     = '1deg_mmean_means_1986-2019'
norm_SLA_thres     = 0.3        # Do not consider SLA values < threshold, set to 0 to deactivate
load_preproc_file  = True       # Skip aggregation to 1° grid, load data from previous run
save_preproc_file  = True       # Save aggregated file, set file name in output_file below
aggregated_input   = False

# Path setup
input_path_rgi     = './data/RGI/'
input_path_tslprep = './data/MANTRA/preprocessed/'
input_path_tslaggr = './data/MANTRA/aggregated/'
output_path_plots  = '<OUTPUT_PATH_PLOTS>'
output_path_tables = '<OUTPUT_PATH_TABLES>'

Path(output_path_plots).mkdir(parents=True, exist_ok=True)
Path(output_path_tables).mkdir(parents=True, exist_ok=True)

# File setup
input_file_tsl     = '<YOUR_MANTRA_RESULT_FILE>.h5'
input_file_grid    = '<YOUR_GRIDDED_RESULT_FILE>.npy'
output_file        = '_grid_export_.npy'
output_file_pc_plot = 'PC_timeseries'
rgi_file           = 'rgi60_Asia.csv'

output_file_name   = 'TSLA-anomaly-abs-'





# Some functions ...
def make_subset_by_months(df, min_month, max_month):
    subset_rule = np.where((df.index.month >= min_month) & (df.index.month <= max_month))[0]
    subset = df.iloc[subset_rule,:]
    return subset.copy()

def calculate_anomaly(df, column_name, agg_column=None):
    if agg_column is not None:
        df = df.groupby(agg_column).mean()
    col_mean = df[column_name].mean()
    df['anomaly'] = df[column_name] - col_mean
    return df


# Import data

if aggregated_input:
    # Import aggregated TSLA data
    print('\nReading aggregated TSLA data. This may take a while ...')
    df_tsl = pd.read_hdf(input_path_tslaggr + input_file_tsl,
                         parse_dates=True,
                         index_col='LS_DATE',
                         low_memory=False)
    # , columns=['RGI_ID', 'SC_median', 'TSL_normalized', 'LS_DATE']
else:
    # Import TSLA data
    print('\nReading TSLA file. This may take a while ...')
    df_tsl = pd.read_hdf(input_path_tslprep + input_file_tsl,
                         parse_dates=True,
                         columns=['RGI_ID', 'SC_median', 'TSL_normalized', 'LS_DATE'],
                         index_col='LS_DATE', low_memory=False)
        
    # Prepare seasonal subsets
    print('Extracting ablation phase subset.')
    df_tsl = make_subset_by_months(df_tsl, 7, 10)
    # winter_subset = make_subset_by_months(df_tsl, 3, 11)

print('Success. Data frame contains '+ str(df_tsl.shape[0]) +' rows and '+
      str(df_tsl.shape[1]) +' columns. \n')   

if norm_SLA_thres > 0:
    SLA_thres_query = np.where((df_tsl['TSL_normalized'] >= norm_SLA_thres))[0]    
    print('Keeping '+ str(len(SLA_thres_query)) +' observations > SLA threshold ('+ str(norm_SLA_thres) +')')
    df_tsl = df_tsl.iloc[SLA_thres_query, :]
    
n_obs_orig = df_tsl.shape[0]
n_glaciers_orig = len(df_tsl['RGI_ID'].unique())

if 'LS_DATE' not in df_tsl.columns:
    df_tsl['LS_DATE'] = df_tsl.index.to_series().dt.strftime('%Y-%m-%d')

df_tsl['date'] = df_tsl['LS_DATE']
df_tsl.index = pd.to_datetime(df_tsl['LS_DATE'])


# Import RGI data
df_rgi = pd.read_csv(input_path_rgi + rgi_file, index_col='RGIId', parse_dates=True, low_memory=False)
# df_rgi_sorted = df_rgi.sort_values(by='Area', ascending=False)
df_rgi['RGI_ID'] = df_rgi.index
df_rgi.rename(columns={'FiTRegionCode': 'region_code', 'FiTRegionLabel': 'region_name', 'CenLon': 'longitude', 'CenLat': 'latitude', 'Area': 'area'}, inplace=True)
df_rgi['elev_range'] = df_rgi['Zmax'] - df_rgi['Zmin']

df_rgi_gt_0_5skm = df_rgi.copy()
df_rgi_gt_0_5skm = df_rgi_gt_0_5skm.iloc[np.where(df_rgi['area'] > 0.5)[0],:]


# Create minimal RGI dataframe
rgi_cols = list(df_rgi.columns)
del_cols = [x for x in rgi_cols if x not in ['region_code', 'region_name', 'RGI_ID', 'longitude', 'latitude', 'area']]
df_rgi_mini = df_rgi.drop(columns=del_cols)


# Merge dataframe with RGI data
df_tsl = pd.merge(df_tsl, df_rgi_mini, how='left', on='RGI_ID')



'''
CONCEPT

## 01 Data Preparation 
- Calulate mean/median normalized TSLA for each 0.2° (?) grid cell and season of year
    1 Meteor. season: 		        DJF, MAM, JJA, SON or
    2 adapted to glacial cylcle	JFM, AMJ, JAS, OND or
    3 winter vs. summer		    NDJFM, JJAS
- Calculate anomalies by removing the time-mean
  (alternatively the HMA-wide spatial mean for the time step?)
  
'''

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
data_column = 'TSL_normalized'



if load_preproc_file:
    print('\nLoading preprocessed file from \n'+ str(input_path_tslprep + input_file_grid))
    grid = np.load(input_path_tslaggr + input_file_grid)
    


#vec_lttas = np.vectorize(loop_through_time_and_space)
#grid = vec_lttas(df_subset, years_unique, grid_lat, grid_lon)
    


# Save grid to file
if save_preproc_file:
    np.save(input_path_tslaggr + output_file, grid)



# Compute temporal anomalies by removing the time-mean.
tsla_mean = np.nanmean(grid, axis=0)
tsla_anomaly = np.nan_to_num(grid - tsla_mean)

# Compute total anomalies by remomving the total mean
tsla_spatiotemp_mean = np.nanmean(tsla_mean)
tsla_total_anomaly = np.nan_to_num(tsla_mean -  tsla_spatiotemp_mean)


tsla_annual_means = [np.nanmean(grid[i,:,:]) for i in range(len(grid[:,0,0]))]
tsla_annual_anomalies = [np.nanmean(tsla_anomaly[i,:,:]) for i in range(len(tsla_anomaly[:,0,0]))]

# for i_year in range(tsla_anomaly[:,0,0]):

anomaly_array = tsla_anomaly

# Prepare the anomaly dataframe
df_anomaly = annual_lat_lon_grid(df_tsl, df_rgi_gt_0_5skm, anomaly_array, start_year=start_year, end_year=end_year, total_values=tsla_total_anomaly)

elev_mean = np.nanmean(df_anomaly['elev_range'])
# tsla_annual_anomalies_m = [ i * elev_mean for i in tsla_annual_anomalies]



# Prepare anomaly time series data
tsla_anomaly_nan = tsla_anomaly.copy()
tsla_anomaly_nan[tsla_anomaly_nan == 0] = np.nan

df_annual_anomaly_grid = pd.DataFrame(columns=['lat', 'lon', 'n_glaciers', 'mean_elev_range', 'anomalies', 'anomalies_m'])
n_glaciers_grid = 0
for lat_id in range(len(grid_lat)):
    for lon_id in range(len(grid_lon)):
        if np.nansum(tsla_anomaly_nan[:,lat_id,lon_id]) != 0:
            
            subset_rgi = df_rgi_gt_0_5skm[(df_rgi_gt_0_5skm['latitude'] >= grid_lat[lat_id] - 1) &
                                          (df_rgi_gt_0_5skm['latitude'] < grid_lat[lat_id]) &
                                          (df_rgi_gt_0_5skm['longitude'] >= grid_lon[lon_id] - 1)  &
                                          (df_rgi_gt_0_5skm['longitude'] < grid_lon[lon_id])]
            
            print('Lat-'+ str(lat_id) +'/Lon-'+ str(lon_id) +': '+ str(grid_lat[lat_id]) +'/'+ str(grid_lon[lon_id]) +' contains '+ str(len(subset_rgi)) +' glaciers' )
            
            cell_mean_elev_rng = subset_rgi['elev_range'].mean()
            cell_anomalies_nrm = list(tsla_anomaly_nan[:, lat_id, lon_id])
            cell_anomalies_abs = [ x * cell_mean_elev_rng for x in cell_anomalies_nrm]
            
            df_annual_anomaly_grid = df_annual_anomaly_grid.append({
                'lat': grid_lat[lat_id], 
                'lon': grid_lon[lon_id], 
                'n_glaciers': len(subset_rgi),
                'mean_elev_range': cell_mean_elev_rng,
                'anomalies': cell_anomalies_nrm,
                'anomalies_m': cell_anomalies_abs}, 
                ignore_index=True)
            
            n_glaciers_grid += len(subset_rgi)

print('n glaciers in grid: '+ str(n_glaciers_grid))

# Add weights based on n_glaciers in cell in relation to total n glaciers
df_annual_anomaly_grid['weight'] = df_annual_anomaly_grid['n_glaciers'] / n_glaciers_grid

# df_tmp = pd.DataFrame(columns=['anomalies_weighted', 'anomalies_m_weighted'])
anomalies_weighted = np.zeros((len(df_annual_anomaly_grid), len(years_unique)))
anomalies_m_weighted = np.zeros((len(df_annual_anomaly_grid), len(years_unique)))
i = 0
for row in df_annual_anomaly_grid.iterrows():
    anomalies_weighted[i] = [ x * row[1]['weight'] for x in row[1]['anomalies']]
    anomalies_m_weighted[i] = [ x * row[1]['weight'] for x in row[1]['anomalies_m']]
    # df_tmp = df_tmp.append({'anomalies_weighted': anomalies_weigthed, 'anomalies_m_weighted': anomalies_m_weigthed}, ignore_index=True)
    i += 1

# for i in range(len(years_unique)):

tsla_annual_anomalies = np.nansum(anomalies_weighted, axis=0)     
tsla_annual_anomalies_m = np.nansum(anomalies_m_weighted, axis=0)    
    
anomaly_series = df_annual_anomaly_grid['anomalies']
anomaly_series_m = df_annual_anomaly_grid['anomalies_m']

print('\nPreprocessing finished.\n')


# Check for trend in annual mean anomalies
print('\n Preparing some stats ...')

from scipy import stats
trend = stats.linregress(years_unique, tsla_annual_anomalies_m)

mktrend = stats.kendalltau(years_unique, tsla_annual_anomalies_m, nan_policy='omit', method='auto')
trend_r2 = trend.rvalue * trend.rvalue


# Calculate 95% confidence interval on slope and intercept:
# Two-sided inverse Students t-distribution
# p - probability, df - degrees of freedom
from scipy.stats import t
tinv = lambda p, df: abs(t.ppf(p/2, df))

ts = tinv(0.05, len(years_unique)-2)
"""
print(f"slope (95%): {trend.slope:.6f} +/- {ts*trend.stderr:.6f}")


print(f"intercept (95%): {trend.intercept:.6f}"
      f" +/- {ts*trend.intercept_stderr:.6f}")
"""

trend_1992_2015 = stats.linregress(years_unique[6:-4], tsla_annual_anomalies_m[6:-4])
mktrand_1992_2015 = stats.kendalltau(years_unique[6:-4], tsla_annual_anomalies_m[6:-4])
trend_1992_2015_r2 = trend_1992_2015.rvalue * trend_1992_2015.rvalue

print('Average annual TSLA rise: '+ str(np.round(trend.slope, 3)) +' +/- '+ str(np.round(ts*trend.stderr, 3)) +' m (R^2 = '+ str(np.round(trend_r2, 3)) +')')
print('Average annual TSLA rise for 1992 to 2015: '+ str(trend_1992_2015.slope) +' +/- '+ str(ts*trend_1992_2015.stderr)+' m (R^2='+ str(trend_1992_2015_r2) +')')



# Prepare maps for individual years
print('\nPreparing plots ...')
for year in years_unique:
    plot_grid_map(df_anomaly, year,  
                  plot_legend=False, 
                  fig_title = str(year),
                  vmin=-200, vmax=200, 
                  bg_img_file = '/home/loibldav/Git/1_TopoCliF/fit-intern/data/DEM/GTOPO30_HMA_0_04.tif',
                  # color_map=SCM6.roma,
                  dpi=dpi, 
                  scaling_factor = 0.6,
                  scaling_base = 10,
                  output_file_basename=output_path_plots + output_file_name, 
                  labels={
                      # 'legend_title': 'n timeseries',
                      'cbar_label': 'Transient snowline altitude anomaly (m)'})



# Prepare maps for time spans
temporal_bins = np.linspace(bin_start_year, bin_start_year + (bin_size * (n_bins - 1)), n_bins, dtype=int)

for tempopral_bin in temporal_bins:
    # five_year_subset_query = np.where((df_anomaly.year >= tempopral_bin) & (df_anomaly.year < tempopral_bin + bin_size))[0]
    # anomaly_temporal_subset = df_anomaly.iloc[five_year_subset_query, :]
    
    # plot_anomaly_map(tempopral_bin, tempopral_bin + bin_size - 1)
    plot_grid_map(df_anomaly, tempopral_bin, tempopral_bin + bin_size - 1, 
                  plot_legend=True, 
                  fig_title = str(tempopral_bin) +' - '+ str(tempopral_bin + bin_size - 1),
                  vmin=vmin, vmax=vmax, 
                  bg_img_file = '/home/loibldav/Git/1_TopoCliF/fit-intern/data/DEM/GTOPO30_HMA_0_04.tif',
                  # color_map=SCM6.roma,
                  scaling_base = 10,
                  scaling_factor = 0.7,
                  dpi=dpi, 
                  output_file_basename=output_path_plots + 'legend_'+ output_file_name, 
                  labels={
                      # 'legend_title': 'n timeseries',
                      'cbar_label': 'Transient snowline altitude anomaly (m)'})







# Prepare time series of AOI-wide mean anomalies (normalized)

plot_anomaly_timeseries(tsla_annual_anomalies, anomaly_series, years_unique,                                   
                  output_file_basename=output_path_plots + output_file_name +'_total_anomaly_timeseries_normalized_', 
                  labels={
                      'x_label': '', #Year C.E.
                      'y_label': 'Normalized transient snowline altitude anomaly'})

# Prepare time series of AOI-wide mean anomalies (absolute in m)



plot_anomaly_timeseries(tsla_annual_anomalies_m, anomaly_series_m, years_unique,                                   
                  output_file_basename=output_path_plots + output_file_name +'_total_anomaly_timeseries_absolute_m', 
                  labels={
                      'x_label': '', #Year C.E.
                      'y_label': 'Transient snowline altitude anomaly (m)'})



end_time = datetime.datetime.now().timestamp()
proc_time = end_time - start_time

print('\nProcessing finished in '+ str(datetime.timedelta(seconds=proc_time)) +'\n')




