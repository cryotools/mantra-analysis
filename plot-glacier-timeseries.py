#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot TSL timeseries for an individual glacier.

@author: David Loibl, Inge Grünberg
"""
 
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import cm
import pandas as pd
# from pandas.plotting import register_matplotlib_converters
import sys
import os

# register_matplotlib_converters()

# USER VARIABLES
verbosity = 2           # Reporting level
plot_dpi = 300
                  

if verbosity >= 2:
    i = 0
    for i in range(len(sys.argv)):
        print('Arg '+ str(i) +' set to '+ str(sys.argv[i]))
        i += 1
    
if len(sys.argv) != 6:
    print("\nUsage: python plot-glacier-timeseries.py <tsl_file> <glacier_list_file> <lower_limit> <upper_limit> <output_dir>\n\n")
    print("   tsl_file             -> A TSL result file in HDF format. ")
    print("   glacier_list_file    -> An ASCII text file containing the RGI_IDs, one ID per row")
    print("   lower_limit          -> First dataset to process, referring to number of IDs in glacier_list_file")
    print("   upper_limit          -> Last dataset to process, referring to number of IDs in glacier_list_file")
    print("   output_dir           -> A valid directory to which the output will be written.\n\n")
    sys.exit(1)


# INPUT FILE
input_file = sys.argv[1]
exists = os.path.isfile(input_file)
if not exists:
    # Use existing glacier ID file to determine glaciers to process ...
    print('\nCRITICAL ERROR')
    print('No input file found at '+ str(input_file))
    print('Exiting ...')
    sys.exit(1)

lower_limit = int(sys.argv[3])
upper_limit = int(sys.argv[4])

# GLACIER LIST
glacier_list_file = sys.argv[2]
exists = os.path.isfile(glacier_list_file)
if not exists:
    # Use existing glacier ID file to determine glaciers to process ...
    print('\nCRITICAL ERROR')
    print('No glacier list file found at '+ str(glacier_list_file))
    print('Exiting ...')
    sys.exit(1)
else:    
    # Use existing glacier ID file to determine glaciers to process ...
    glacier_ids = []
    
    with open(glacier_list_file, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            current_glacier = line[:-1]
    
            # add item to the list
            glacier_ids.append(current_glacier)    
            
    glacier_ids = glacier_ids[lower_limit:upper_limit]
    
    
#glacier_id_string = sys.argv[2].replace('"', '').replace('\n', '').replace(' ', '').replace("'", '')
#glacier_ids = glacier_id_string.split(",")

# OUTPUT DIRECTORY
output_dir = sys.argv[5]


# print('glacier_ids: '+ str(glacier_ids))

print('\nReading input file. This may take a while ...')
df_TSL = pd.read_hdf(input_file, where=['RGI_ID in glacier_ids'], parse_dates=True, index_col='LS_DATE', low_memory=False)

print('Success. Data frame contains '+ str(df_TSL.shape[0]) +' rows and '+ str(df_TSL.shape[1]) +'columns. \n')


# print(str(df_TSL.head()))
# glacier_ids = df_TSL.RGI_ID.unique()
# np.savetxt(output_dir +'glaciers.list', glacier_ids, delimiter='\n', fmt='%s')

df_TSL.index = pd.to_datetime(df_TSL.LS_DATE, format='%Y-%m-%d')
print('\nColumns: '+ str(df_TSL.columns))
#glacier_ids = glacier_ids[10000:10010]


for glacier_id in glacier_ids:
    print('\nPreparing TSL time series for '+ str(glacier_id))
    
    current_glacier = df_TSL.loc[df_TSL.RGI_ID == glacier_id, :]
    
    print('Current glacier: '+ str(current_glacier.RGI_ID) +' has '+ str(current_glacier.shape[0]) +' rows and '+ str(current_glacier.shape[1]) +' cols')
    
    years = current_glacier['LS_DATE'].values.astype('datetime64[Y]')
    current_glacier.index = pd.to_datetime(current_glacier.LS_DATE, format='%Y-%m-%d')
    
    # find July, August, September and October points
    jas = np.where((current_glacier.index.month >= 7) & (current_glacier.index.month < 10))[0]
    print('\nn summer obs: ' + str(len(jas)) +'\n')
    
    """
    # annual maximum
    anmax = current_glacier2.SC_median.resample('AS').max()
    print('anmax: ' + str(anmax))
    # locations of the annual maxima
    idx = current_glacier2.SC_median.resample('A').agg(lambda x: np.nan if x.count() == 0 else x.idxmax())
    anmax.index = idx
    """
    '''
    # Annual maxima
    # RG_series = current_glacier.copy()
    current_glacier.index = pd.to_datetime(current_glacier.LS_DATE, format='%Y-%m-%d')
    max_idx = current_glacier.groupby(current_glacier.index.year)['SC_median'].transform(max) == current_glacier['SC_median']
    # print('max_idx :' + str(max_idx))
    # winter_max_idx2 = current_glacier.groupby([current_glacier.index.year]).max()
    # print('winter_max_idx 2:' + str(winter_max_idx2))
    annual_maxima = current_glacier[max_idx]
    '''
    # annual maximum
    anmax = current_glacier.SC_median.resample('AS').max()
    # locations of the annual maxima
    idx = current_glacier.SC_median.resample('A').agg(lambda x : np.nan if x.count() == 0 else x.idxmax())
    anmax.index = idx
    
    fig = plt.figure(figsize=(3000/plot_dpi, 1500/plot_dpi), dpi=plot_dpi)
    
    # first plot
    ax = fig.add_subplot(111)
        
    for xc in years:
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
    fig_name = str(glacier_id)+'.png'
    print('Saving figure ' + fig_name)
    plt.savefig(output_dir + fig_name, bbox_inches='tight')
    plt.close()

print('\nProcessing finished.\n')


'''
for i1 in range(n):
    
    
    for j2 in range(n2):
        i2 = glacier_ids[i[j2]]
        print('Adding glacier ' + str(i2))
        # create small time series for testing
        current_glacier = df_TSL.loc[df_TSL.RGI_ID == i2, :]
        #current_glacier.reset_index(inplace=True)
        # rg_stats = str(current_glacier.head())
        # print(rg_stats + '\n')

        years = current_glacier['LS_DATE'].values.astype('datetime64[Y]')
        current_glacier.index = pd.to_datetime(current_glacier.LS_DATE, format='%Y-%m-%d')
        # take only OK points
        current_glacier2 = df_TSL.loc[df_TSL.RGI_ID == i2, :]
        current_glacier2.index = pd.to_datetime(current_glacier2.LS_DATE, format='%Y-%m-%d')

        # current_glacier2 = current_glacier.iloc[np.where(current_glacier.quality == True)[0],:]
        # find July, August, September and October points
        jas = np.where((current_glacier2.index.month >= 7) & (current_glacier2.index.month <= 10))[0]

        """
        # annual maximum
        anmax = current_glacier2.SC_median.resample('AS').max()
        print('anmax: ' + str(anmax))
        # locations of the annual maxima
        idx = current_glacier2.SC_median.resample('A').agg(lambda x: np.nan if x.count() == 0 else x.idxmax())
        anmax.index = idx
        """

        # Annual maxima
        RG_series = current_glacier2.copy()
        RG_series.index = pd.to_datetime(RG_series.LS_DATE, format='%Y-%m-%d')
        max_idx = RG_series.groupby(RG_series.index.year)['SC_median'].transform(max) == RG_series['SC_median']
        # print('max_idx :' + str(max_idx))
        # winter_max_idx2 = RG_series.groupby([RG_series.index.year]).max()
        # print('winter_max_idx 2:' + str(winter_max_idx2))
        annual_maxima = RG_series[max_idx]
        # print('Annual maxima: ' + str(annual_maxima))

        # first plot
        ax = fig.add_subplot(5, 2, j2+1)

        # vertical lines for each year
        for xc in years:
            ax.axvline(x=xc, color='0.8', linestyle=':', linewidth=0.2)

        # grey dots for suspicious values
        ax.plot(current_glacier.SC_median, '0.7', marker='.', linestyle='')
        ax.plot(current_glacier.SC_median, '0.7', marker='', linestyle='-', linewidth=0.1)

        # Blue dots + thin line for ok values
        ax.plot(current_glacier2.SC_median, 'b.-', linewidth=0.1)

        # green dots for summer values
        ax.plot(current_glacier2.SC_median[jas], 'g.')

        # label with the glacier name
        ax.text(0.01, 0.9, i2, transform=ax.transAxes)
        ax.text(0.01, 0.8, '(' + str(round(current_glacier2.glacier_area[0], 2)) + ' km²)', transform=ax.transAxes)

        if j2 == 0:
            ax.set_title('Transient snowline time series')
    fig_name = 'Set'+str(i1)+'_seperate.pdf'
    print('Saving figure ' + fig_name)
    fig.savefig(plot_dir + fig_name, bbox_inches='tight')
    plt.close(fig)

'''
