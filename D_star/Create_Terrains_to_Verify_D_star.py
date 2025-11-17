#%% Script to create artificial terrains used to set D*
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
from landlab import imshow_grid
from landlab.components import StreamPowerEroder, DepressionFinderAndRouter, LinearDiffuser, FlowAccumulator 
from landlab.io.esri_ascii import read_esri_ascii, write_esri_ascii
from matplotlib import pyplot as plt
from osgeo import gdal

# Section break
print(' ')

#%% Set directories

# Print status
print('Setting directories...')

# Set directories
script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_directory = script_directory.replace('\d_prime', '')
export_directory = script_directory + '/'

# Create directories
print('Creating directory...')
if path.exists(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output') == False:
    os.mkdir(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output')

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
K_sp = 5E-7   

# Diffusivity (change this between runs to create unique terrains)
D_star = 375
K_hs = K_sp * D_star                                          

# 10Be mean catchment erosion rate, assumed to be uplift rate
SAP56_mean_recalc = 7.96 * 1E-6          # m / yr^-1

# Set time parameters
dt = 1E6                    # years
tmax = 1E9                  # years
t = np.arange(0, tmax, dt) 

# Set clocks
total_time = 0 
plot_ticker = 0
export_DEM_ticker = 0

# Export parameters
plot_interval = 1E8         # years
export_DEM_interval = 1E9   # years
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
previous_zr = mg.add_zeros('node', 'previous_topographic__elevation')
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
 
# Initialize previous_zr
previous_zr[mg.core_nodes] = zr[mg.core_nodes]

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
    
    # Calculate dz_dt
    dz_dt[mg.core_nodes] = (zr[mg.core_nodes] - previous_zr[mg.core_nodes]) / dt
    
    # Record current topograpghy for comparrison before and after timestep
    previous_zr[mg.core_nodes] = zr[mg.core_nodes]    
    
    # Update time
    total_time += dt

    # Print time                                    
    print('Time = ', total_time)
    
    # Update clocks
    plot_ticker += dt
    export_DEM_ticker += dt
    
    # Plots
    if plot_ticker == plot_interval:

        # Print status
        print('Exporting plots...')
        
        # DEM_Image
        if path.exists(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/DEM_Image') == False:
            os.mkdir(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/DEM_Image')
        plt.ioff()
        fig = plt.figure(1)         
        imshow_grid(mg, 'topographic__elevation', grid_units=('m', 'm'), var_name="Elevation (m)", cmap='terrain', allow_colorbar=True)
        title_text = '$Year$='+str(total_time)  
        plt.title(title_text)
        plt.tight_layout()
        fig.savefig(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/DEM_Image/'+str(total_time)+'.'+export_format,  format=export_format, dpi=dpi)
        plt.close(fig)
        
        # DZ_DT_Map
        if path.exists(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/DZDT_Map') == False:
            os.mkdir(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/DZDT_Map')
        plt.ioff()
        fig = plt.figure(8)
        imshow_grid(mg, "elevational_change", grid_units=("m", "m"), var_name="Rate of Elevational Change (m/yr)", cmap="seismic_r", symmetric_cbar = True)
        title_text = '$Year$='+str(total_time)  
        plt.title(title_text)
        plt.tight_layout()
        fig.savefig(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/DZDT_Map/'+str(total_time)+'.'+export_format,  format=export_format, dpi=dpi)
        plt.close(fig)
            
        # Reset plot_ticker    
        plot_ticker = 0
            
    # Export DEM, continuation       
    if export_DEM_ticker == export_DEM_interval:
        
        # Print status
        print('Exporting DEM...')
        
        # Export
        if path.exists(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Export_DEM') == False:
            os.mkdir(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Export_DEM')
        write_esri_ascii(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Export_DEM/'+str(total_time)+'.asc', mg, names='topographic__elevation', clobber = True)
        ds = gdal.Open(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Export_DEM/'+str(total_time)+'.asc')
        ds = gdal.Warp(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Export_DEM/'+str(total_time)+'.tif', ds, format="GTiff", dstSRS="EPSG:32616")
        os.remove(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Export_DEM/'+str(total_time)+'.asc')
        
        # Reset export_DEM_ticker
        export_DEM_ticker = 0

# Section break        
print('')

#%% Export final grid

# Print status    
print('Exporting final grid...')

# Export
if path.exists(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Final_DEMs') == False:
    os.mkdir(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Final_DEMs')
write_esri_ascii(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Final_DEMs/'+str(D_star)+'.asc', mg, names='topographic__elevation', clobber = True)
ds = gdal.Open(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Final_DEMs/'+str(D_star)+'.asc')
ds = gdal.Warp(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Final_DEMs/'+str(D_star)+'.tif', ds, format="GTiff", dstSRS="EPSG:32616")
os.remove(str(export_directory)+'/Create_Terrains_to_Verify_D_prime_output/Final_DEMs/'+str(D_star)+'.asc')
   
# Section break        
print('')
     
#%% Wrap-up

# Print status    
print('Complete!')