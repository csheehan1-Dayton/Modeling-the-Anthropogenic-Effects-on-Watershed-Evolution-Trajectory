#%% Run K0 Deep Low
# **Author**: Chris Sheehan

#%% Import libraries

# Print status
print('Importing libraries...')

# Import
import os
import os.path
from os import path
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
import inspect
import numpy as np
from numpy import nan
import pandas as pd
from landlab import imshow_grid
from landlab.components import StreamPowerEroder, DepressionFinderAndRouter, LinearDiffuser, FlowAccumulator 
from landlab.io.esri_ascii import read_esri_ascii, write_esri_ascii
from matplotlib import pyplot as plt
import matplotlib.cbook as cbook
from osgeo import gdal

# Section break
print(' ')

#%% Set directories

# Print status
print('Setting directories...')

# Set directories
script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
script_name = os.path.basename(sys.argv[0])
run_name = script_name.replace('.py', '')
project_directory = script_directory.replace('\\runs\\' + run_name, '')
EET_directory = project_directory + '\\EET'
export_directory = script_directory + '/Output/'

# Create directories
if path.exists(str(script_directory)+'/Output') == False:
    os.mkdir(str(script_directory)+'/Output')
    
# Import EET
sys.path.append(EET_directory)
import EET

# Section break
print(' ')

#%% Parameters

# Print status
print('Setting parameters...')

# Grid parameters
DEM_path = project_directory + '\Input\Chestatee_model_grid.asc'    # grid path
no_data_value = -99999                                              # grid no data value  
pour_point_node = 327                                               # watershed outlet node                 

# Stream power parameters
m_sp = 0.503                         
n_sp = 1.224                          
K_sp = 1.01349004571795E-06    

# Diffusivity (change this between runs to create unique terrains)
D_star = 262.5
K_hs = K_sp * D_star                                          

# 10Be mean catchment erosion rate, assumed to be uplift rate
SAP56_mean_recalc = 7.96 * 1E-6          # m / yr^-1

# Set time parameters
dt = 100                    # years
tmax = 1E6                  # years
t = np.arange(0, tmax, dt) 

# Set clocks
total_time = 0 
plot_ticker = 0
export_DEM_ticker = 0
timestep_integer = 0

# Export parameters
plot_interval = 1E5          # years
export_DEM_interval = 1E5     # years
export_format = 'png'
dpi = 300 

# Section break
print(' ')

#%% Initialize grid and associated parameters

# Print status
print('Initializing grid...')

# Set random seed
np.random.seed(0) 

# Import DEM
(mg, zr) = read_esri_ascii(DEM_path, name='topographic__elevation')

# Handle non-value nodes
mg.set_nodata_nodes_to_closed(zr, no_data_value)

# Create node keys
dz_dt = mg.add_zeros('node', 'elevational_change')

# Handle Grid Boundaries
mg.set_status_at_node_on_edges(right=4, top=4, left=4, bottom=4)
mg.status_at_node[pour_point_node] = mg.BC_NODE_IS_FIXED_VALUE

# Set stream power and uplift paramters across grid
K = np.ones(mg.number_of_nodes) * K_sp
U = np.ones(mg.number_of_nodes) * SAP56_mean_recalc

# Section break
print(' ')

#%% Initialize components

# Print status
print('Initializing components...')

# Initialize FlowAccumulator
frr = FlowAccumulator(mg, flow_director='D8')
frr.run_one_step()

# Initialize LinearDiffuser
dfn = LinearDiffuser(mg, linear_diffusivity = K_hs, method = 'simple', deposit = False)        

# Initialize StreamPowerEroder 
spr = StreamPowerEroder(mg, K_sp = K, m_sp = m_sp, n_sp = n_sp, erode_flooded_nodes = True)

# Initialize DepressionFinderAndRouter
dfr = DepressionFinderAndRouter(mg)

# # Initialize ErosionElevationTracker
eet = EET.ErosionElevationTracker(mg, bedrock__and__soil = False)

# Initialize appendable arrays and lists
timesteps = []
times = []
mean_basin_erosion_rate = []
sum_basin_erosion = []
stats_list = list()

# Section break
print(' ')

#%% Time Loop

# Print status
print('Starting time loop...')

# Time loop
for ti in t:
      
    # Uplift topograpghy
    zr[mg.core_nodes] += U[mg.core_nodes]*dt    
    
    # Run one steps                       
    frr.run_one_step()  
    dfr.map_depressions()                                  
    spr.run_one_step(dt)
    dfn.run_one_step(dt)
    eet.run_one_step(dt, uplift = SAP56_mean_recalc * dt)
    
    # Calculate dzdt and erosion rate
    dzdt = eet.return_dzdt()
    erosionrate = eet.return_erosionrate() 
    
    # Adjust dzdt and erosion rate for no data nodes
    dzdt[np.where(zr == no_data_value)] = nan
    erosionrate[np.where(zr == no_data_value)] = nan
    
    # Update appendable arrays and lists
    timesteps = np.append(timesteps, timestep_integer)
    times = timesteps * dt
    mean_basin_erosion_rate = np.append(mean_basin_erosion_rate, np.nanmean(erosionrate))
    sum_basin_erosion = np.append(sum_basin_erosion, np.nansum(eet.return_erosion()))
    labels = list()
    label = total_time + dt
    label = str(label)
    labels.append(label)
    erosionrate_f = erosionrate[~np.isnan(erosionrate)]
    stats = cbook.boxplot_stats(erosionrate_f, labels=labels)
    stats = stats[0]
    stats_list.append(stats)

    # Update time
    total_time += dt
    total_time = np.round(total_time, 5)

    # Print time                                    
    print('Time = ', total_time)
    
    # Update clocks
    plot_ticker += dt
    plot_ticker = np.round(plot_ticker, 5)
    export_DEM_ticker += dt
    export_DEM_ticker = np.round(export_DEM_ticker, 5)
    timestep_integer += 1
    
    # Plots
    if plot_ticker >= plot_interval:

        # Print status
        print('Exporting plots...')
        
        # DEM_Image
        if path.exists(str(export_directory)+'DEM_Image') == False:
            os.mkdir(str(export_directory)+'DEM_Image')
        plt.ioff()
        fig = plt.figure()         
        imshow_grid(mg, 'topographic__elevation', grid_units=('m', 'm'), var_name="Elevation (m)", cmap='terrain', allow_colorbar=True)
        title_text = '$Year$='+str(np.round(total_time))  
        plt.title(title_text)
        plt.tight_layout()
        fig.savefig(str(export_directory)+'DEM_Image/'+str(np.round(total_time))+'.'+export_format,  format=export_format, dpi=dpi)
        plt.close(fig)
        
        # DZ_DT_Map
        if path.exists(str(export_directory)+'DZDT_Map') == False:
            os.mkdir(str(export_directory)+'DZDT_Map')
        plt.ioff()
        fig = plt.figure()
        imshow_grid(mg, dzdt, grid_units=("m", "m"), var_name="Rate of Elevational Change (m/yr)", cmap="seismic_r", symmetric_cbar = True, vmin = -np.nanpercentile(dzdt, 99), vmax = np.nanpercentile(dzdt, 99))
        title_text = '$Year$='+str(np.round(total_time))  
        plt.title(title_text)
        plt.tight_layout()
        fig.savefig(str(export_directory)+'DZDT_Map/'+str(np.round(total_time))+'.'+export_format,  format=export_format, dpi=dpi)
        plt.close(fig)
            
        # Reset plot_ticker    
        plot_ticker = 0
            
    # Export DEM, continuation       
    if export_DEM_ticker >= export_DEM_interval:
        
        # Print status
        print('Exporting DEM...')
        
        # Export
        if path.exists(str(export_directory)+'Export_DEM') == False:
            os.mkdir(str(export_directory)+'Export_DEM')
        write_esri_ascii(str(export_directory)+'Export_DEM/'+str(np.round(total_time))+'.asc', mg, names='topographic__elevation', clobber = True)
        ds = gdal.Open(str(export_directory)+'Export_DEM/'+str(np.round(total_time))+'.asc')
        ds = gdal.Warp(str(export_directory)+'Export_DEM/'+str(np.round(total_time))+'.tif', ds, format="GTiff", dstSRS="EPSG:32616")
        os.remove(str(export_directory)+'Export_DEM/'+str(np.round(total_time))+'.asc')
        
        # Reset export_DEM_ticker
        export_DEM_ticker = 0

# Section break        
print('')


#%% Export csvs

# Print status
print('Exporting csvs...')

np.savetxt(str(export_directory) + 'mean_basin_erosion_rate.csv', mean_basin_erosion_rate, delimiter = ",")
np.savetxt(str(export_directory) + 'sum_basin_erosion.csv', sum_basin_erosion, delimiter = ",")
np.savetxt(str(export_directory) + 'times.csv', times, delimiter = ",")
np.savetxt(str(export_directory) + 'timesteps.csv', timesteps, delimiter = ",")
df = pd.DataFrame.from_dict(stats_list)
df.to_csv(str(export_directory) + 'erosion_stats.csv', index = False, header=True)

# Section break        
print('')

#%% Finalize

# Print status
print('Finished!')