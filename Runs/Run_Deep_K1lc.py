#!/usr/bin/env python
# coding: utf-8

# # TerrainSandbox 
# 
# **Author**: Chris Sheehan

#%% Block 1: The Grid

# Import libraries required to import K 
print('Importing libraries...')
import numpy as np
import pandas as pd

import inspect
import os
script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_directory = script_directory.replace('\\runs\\run_deep_k1lc', '')

# ENTER VARIABLES ############################################################
##############################################################################

m_sp = 0.503                         
n_sp = 1.224                         
#K_base = 7.50E-07
#D = (K * 800) 

# Import K 
K_path = project_directory + '\K1_Create\Output\K1.csv'
K = pd.read_csv(K_path, header = None)
K = K.values
K = np.squeeze(K)
D = (K * 262.5) 

DEM_path = project_directory + '\DEM\Chestatee.asc'        # '/Users/Chris/Dropbox/BC_Landlab/SPACE_2022/Chestatee/D_Calibration_New/DEMs/A.asc'
dxy = 26.673674998310        
no_data_value = -99999   
pour_point_node = 327       # pre-selected. Ideal beacause (A). Running  FlowAccumulator on raw DEM identifies this node, and (B). It is a boundary node. Chestatee = 327. A = 3959.

lc_path = project_directory + '/K1_Create/NCLD_2019_Align.asc' 
lc_scheme_path = project_directory + '\K1_Create\Erosion_Rate_Scheme.csv' 

SAP56_mean_recalc = 7.96 * 1E-6          # m / yr^-1

Total_area = 608568641.6459954        # m^2, calculated using flow router

Directory = script_directory + '/'
Export_format = 'png'
dpi = 600

##############################################################################
# ENTER VARIABLES ############################################################

#OPERATORS-->DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE-----------------------------

# Print space
print(' ')

# Import libraries
print('Importing libraries...')
from numpy import nan, isnan
import math
import os
import os.path
from os import path
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
from landlab import RasterModelGrid, imshow_grid
from landlab.plot.video_out import VideoPlotter
from landlab.components import PerronNLDiffuse, StreamPowerEroder, Space, DepressionFinderAndRouter, LinearDiffuser, TaylorNonLinearDiffuser,  FlowAccumulator, ChannelProfiler, SteepnessFinder, ChiFinder, Lithology, LithoLayers, NormalFault
from landlab.io.esri_ascii import write_esri_ascii
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as colors
from landlab.plot.graph import plot_graph
from landlab.io.esri_ascii import read_esri_ascii 
import EET
import matplotlib.cbook as cbook

# Set random seed
print('Setting random seed...')
np.random.seed(0) 

# Create directories
print('Creating directory...')
if path.exists(str(Directory)+'/TerrainSandbox') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox')
if path.exists(str(Directory)+'/TerrainSandbox/Initial_Condition') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Initial_Condition')
if path.exists(str(Directory)+'/TerrainSandbox/Analysis') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Analysis')
if path.exists(str(Directory)+'/TerrainSandbox/CSV') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/CSV')

# Import DEM
print('Importing DEM...')
(mg, zr) = read_esri_ascii(DEM_path, name='topographic__elevation')

# Create copy of DEM
print('Copying DEM for initial topography comparisons...')
(mg0, zr0) = read_esri_ascii(DEM_path, name='topographic__elevation')

# Handle model DEM dimensions and non-value nodes
print('Handling DEM dimensions and non-value nodes...')
mg.set_nodata_nodes_to_closed(zr, no_data_value)
mg0.set_nodata_nodes_to_closed(zr0, no_data_value)
no_data_nodes = np.where(mg.at_node['topographic__elevation'] == no_data_value)
no_data_nodes = no_data_nodes[0]
number_of_rows = mg.number_of_cell_rows
number_of_columns = mg.number_of_cell_columns
nodes = np.arange(0, np.size(zr))

# Create node keys
print('Creating node keys...')
mg.add_zeros('node', 'soil__flux')

# Handle Grid Boundaries
print('Handling grid boundaries...')
mg.set_status_at_node_on_edges(right=4, top=4, left=4, bottom=4)
mg.status_at_node[pour_point_node] = mg.BC_NODE_IS_FIXED_VALUE

# Enforce domain as a single watershed
print('Enforcing domain as a single watershed...')
number_of_watersheds = 1

# Record pour point node elevation
print('Identifying pour point...')
pour_point_node_elevation = zr[pour_point_node]

# Set uplift rate
print('Setting uplift rate...')
U = np.ones(mg.number_of_nodes) * SAP56_mean_recalc

# Initialize FlowAccumulator
print('Initializing FlowAccumulator...')
frr = FlowAccumulator(mg, flow_director='D8')
frr.run_one_step()

# Plot initial model terrain
print('Plotting initial model terrain...')
if path.exists(str(Directory)+'/TerrainSandbox/Initial_Condition') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Initial_Condition')
plt.ioff()
fig = plt.figure()         
imshow_grid(mg, 'topographic__elevation', grid_units=('m', 'm'), var_name="Elevation (m)", cmap='terrain', allow_colorbar=True)
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/Initial_Condition/DEM_Image.' + Export_format, dpi = dpi)
plt.close(fig)

# Plot initial channel network
print('Plotting initial channel network...')
prf = ChannelProfiler(mg, number_of_watersheds=number_of_watersheds, main_channel_only=False, minimum_channel_threshold=10000000)
prf.run_one_step()
if path.exists(str(Directory)+'/TerrainSandbox/Initial_Condition') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Initial_Condition')
plt.ioff()
fig = plt.figure()
prf.plot_profiles_in_map_view()
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/Initial_Condition/Channel_Network.' + Export_format,  format=Export_format,  dpi=dpi)
plt.close(fig)

# Import Land Cover
print('Importing NLCD data...')
lc = read_esri_ascii(lc_path)
lc = lc[1]

# Import land cover scheme
print('Importing land cover scheme...')
lc_scheme = pd.read_csv(lc_scheme_path)

# Plot land cover
print('Plotting land cover...')
if path.exists(str(Directory)+'/TerrainSandbox/Initial_Condition') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Initial_Condition')
plt.ioff()
fig = plt.figure()
imshow_grid(mg, lc, grid_units=('m', 'm'), var_name="NLCD Catagory", cmap='jet', allow_colorbar=True)
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/Initial_Condition/Land_Cover.' + Export_format,  format=Export_format,  dpi=dpi)
plt.close(fig)

# Plot K
print('Plotting K...')
if path.exists(str(Directory)+'/TerrainSandbox/Initial_Condition') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox/Initial_Condition')
plt.ioff()
fig = plt.figure()
imshow_grid(mg, K, grid_units=('m', 'm'), var_name="Fluvial Erodibility", cmap='plasma', allow_colorbar=True)
plt.tight_layout()
fig.savefig(str(Directory)+'/TerrainSandbox/Initial_Condition/K.' + Export_format,  format=Export_format,  dpi=dpi)
plt.close(fig)

# Print space
print(' ')





#%% BLOCK 5: RESET THE MODEL TIME AND THE PLOTTING TIMERS TO 0

#OPERATORS-->DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE-----------------------------

# Reset
print('Reseting time parameters...')
total_time = 0 
Plot_Ticker = 0
Export_DEM_Ticker = 0
timestep_integer = 0

# Print space
print(' ')





#%% BLOCK 6: TIME PARAMETERS

# ENTER VARIABLES ############################################################
##############################################################################

dt = 100          
tmax = 1E6 

##############################################################################
# ENTER VARIABLES ############################################################

#OPERATORS-->DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE-----------------------------

# Set t
print('Setting t...')
t = np.arange(0, tmax, dt) 

# Print space
print(' ')





#%% BLOCK 7: PLOTTING OPTIONS

# ENTER VARIABLES ############################################################
##############################################################################

# Intervals
Plot_interval = 1E5
Export_DEM_Interval = 1E5

# Drainage information
min_drainage_area = (dxy**2) * 10000
main_channel_only = False

# Toggle maps
DEM_Image = True
Channel_Map = False
Ksn_Map = False
Chi_Map = False
Erosion_Rate_Map = True
DZ_DT_Map = True
Net_Erosion_Map = True
Net_DZ_Map = True

# Toggle profiles
Slope_Area = False
Channel_Profile = False
Ksn_Profile = False
Chi_Profile = False
Erosion_Rate_Profile = False
DZ_DT_Profile = False

# Toggle timeseries
Mean_Basin_Erosion_Rate = True

# Toggle DEM export
Export_DEM = True


##############################################################################
# ENTER VARIABLES ############################################################

#OPERATORS-->DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE-----------------------------

Plot_Ticker = 0
Export_DEM_Ticker = 0

# Notify
print('Setting plot parameters')

# Print space
print(' ')





#%% BLOCK 8: TIME LOOP

#OPERATORS-->DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE-----------------------------

# Create directory
print('Creating directory...')
if path.exists(str(Directory)+'/TerrainSandbox') == False:
    os.mkdir(str(Directory)+'/TerrainSandbox')

# Initialize linear diffuser
print('Initializing LinearDiffuser...') 
dfn = LinearDiffuser(mg, 
                      linear_diffusivity = D, 
                      method = 'simple',
                      deposit = False
                      )        

# Initialize StreamPowerEroder 
print('Initializing StreamPowerEroder...')  
spr = StreamPowerEroder(mg, 
                        K_sp = K, 
                        m_sp = m_sp, 
                        n_sp = n_sp, 
                        erode_flooded_nodes = True
                        )

# Initialize DepressionFinderAndRouter
print('Initializing DepressionFinderAndRouter...') 
dfr = DepressionFinderAndRouter(mg)

# Initialize ErosionElevationTracker
print('Initializing ErosionElevationTracker...') 
eet = EET.ErosionElevationTracker(mg, bedrock__and__soil = False)

# Initialize appendable arrays
print('Initializing appendable arrays') 
mean_basin_erosion_rate = []
sum_basin_erosion = []
timesteps = []
times = []
#
std_basin_erosion_rate = []
stats_list = list()

# Initialize ChannelProfiler
print('Initializing ChannelProfiler') 
if main_channel_only == True:
    prf = ChannelProfiler(mg, number_of_watersheds=number_of_watersheds, main_channel_only=True, minimum_channel_threshold=min_drainage_area)
if main_channel_only == False:
    prf = ChannelProfiler(mg, number_of_watersheds=number_of_watersheds, main_channel_only=False, minimum_channel_threshold=min_drainage_area)

# Print space
print(' ')

# Initialize previous_zr
print('Starting time loop...')  
print(' ')
for ti in t:
      
    # Uplift topograpghy
    zr[mg.core_nodes] += U[mg.core_nodes]*dt    
    
    # Run one steps                       
    frr.run_one_step()  
    dfr.map_depressions()                                  
    spr.run_one_step(dt)
    dfn.run_one_step(dt)
    eet.run_one_step(dt, uplift = U * dt)

    # Calculate dz_dt and erosion rate
    dzdt = eet.return_dzdt()
    erosionrate = eet.return_erosionrate() 
    
    # Update mean_basin_erosion_rate
    #erosionrate[np.where(erosionrate == 0)] = nan
    erosionrate[np.where(zr == no_data_value)] = nan
    mean_basin_erosion_rate = np.append(mean_basin_erosion_rate, np.nanmean(erosionrate))
    sum_basin_erosion = np.append(sum_basin_erosion, np.nansum(eet.return_erosion()))
    timesteps = np.append(timesteps, timestep_integer)
    times = timesteps * dt
    #
    std_basin_erosion_rate = np.append(std_basin_erosion_rate, np.nanstd(erosionrate))
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
    Plot_Ticker += dt
    Export_DEM_Ticker += dt
    timestep_integer += 1
    total_time_r = np.round(total_time, decimals = 1)
    print(total_time_r, ' ', np.max(mg.at_node['drainage_area']))
    
    # Plots
    if Plot_Ticker >= Plot_interval - 0.0000000001:
        print('Exporting figures... Please be patient!')
        
        # Initialize SteepnessFinder and ChiFinder
        if total_time == Plot_Ticker:
            sf = SteepnessFinder(mg, reference_concavity=m_sp/n_sp, min_drainage_area=min_drainage_area) # NEED TO FIX! Currently breaks if you don't export images during first run of Cell 8 but then export them during later runs of Cell 8
            cf_check = 'cf' in locals()
            if cf_check == False:
                cf = ChiFinder(mg, reference_concavity=m_sp/n_sp, min_drainage_area=min_drainage_area)
        
        # DEM_Image
        if DEM_Image == True:
            if path.exists(str(Directory)+'/TerrainSandbox/DEM_Image') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/DEM_Image')
            plt.ioff()
            fig = plt.figure(1)         
            imshow_grid(mg, 'topographic__elevation', grid_units=('m', 'm'), var_name="Elevation (m)", cmap='terrain', allow_colorbar=True)
            title_text = '$Year$='+str(total_time_r)  
            plt.title(title_text)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/DEM_Image/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            plt.close(fig)
        
        # Slope_Area
        if Slope_Area == True:
            prf.run_one_step()
            if path.exists(str(Directory)+'/TerrainSandbox/Slope_Area') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Slope_Area')
            plt.ioff()
            fig = plt.figure(2)
            for i, outlet_id in enumerate(prf.data_structure):
                for j, segment_id in enumerate(prf.data_structure[outlet_id]):
                    if j == 0:
                        label = "channel {i}".format(i=i + 1)
                    else:
                        label = '_nolegend_'
                    segment = prf.data_structure[outlet_id][segment_id]
                    profile_ids = segment["ids"]
                    color = segment["color"]
                    plt.loglog(mg.at_node["drainage_area"][profile_ids][2 : -1], mg.at_node["topographic__steepest_slope"][profile_ids][2 : -1], '.', color=color, label=label)
            plt.legend(loc="lower left")
            plt.xlabel("Drainage Area (m^2)")
            plt.ylabel("Channel Slope [m/m]")
            title_text = '$Year$='+str(total_time_r)  
            plt.title(title_text)
            plt.grid(linestyle='--')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Slope_Area/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            plt.close(fig)
        
        # Channel_Profile
        if Channel_Profile == True:
            prf.run_one_step()
            if path.exists(str(Directory)+'/TerrainSandbox/Channel_Profile') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Channel_Profile')
            plt.ioff()
            fig = plt.figure(3)
            prf.plot_profiles(field='topographic__elevation', xlabel='Distance Upstream (m)', ylabel='Elevation (m)')
            title_text = '$Year$='+str(total_time_r)
            plt.title(title_text)
            plt.grid(linestyle='--')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Channel_Profile/'+str(total_time_r)+'.'+Export_format,  format=Export_format,  dpi=dpi)
            plt.close(fig)
        
        # Channel_Map
        if Channel_Map == True:
            prf.run_one_step()
            if path.exists(str(Directory)+'/TerrainSandbox/Channel_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Channel_Map')
            plt.ioff()
            fig = plt.figure(4)
            prf.plot_profiles_in_map_view()
            title_text = '$Year$='+str(total_time_r)
            plt.title(title_text)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Channel_Map/'+str(total_time_r)+'.'+Export_format,  format=Export_format,  dpi=dpi)
            plt.close(fig)
        
        # Ksn_Profile
        if Ksn_Profile == True:
            prf.run_one_step()
            #sf.calculate_steepnesses()
            if path.exists(str(Directory)+'/TerrainSandbox/Ksn_Profile') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Ksn_Profile')
            plt.ioff()
            fig = plt.figure(5)
            for i, outlet_id in enumerate(prf.data_structure):
                for j, segment_id in enumerate(prf.data_structure[outlet_id]):
                    if j == 0:
                        label = "channel {i}".format(i=i + 1)
                    else:
                        label = '_nolegend_'
                    segment = prf.data_structure[outlet_id][segment_id]
                    profile_ids = segment["ids"]
                    distance_upstream = segment["distances"]
                    color = segment["color"]
                    plt.plot(distance_upstream[2 : -1], mg.at_node["channel__steepness_index"][profile_ids][2 : -1], 'x', color=color, label=label)
            plt.xlabel("Distance Upstream (m)")
            plt.ylabel("Steepness Index")
            plt.legend(loc="upper left")
            title_text = '$Year$='+str(total_time_r)  
            plt.title(title_text)
            plt.grid(linestyle='--')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Ksn_Profile/'+str(total_time_r)+'.'+Export_format,  format=Export_format,  dpi=dpi)
            plt.close(fig)
        
        # Ksn_Map
        if Ksn_Map == True:
            sf.calculate_steepnesses()
            if path.exists(str(Directory)+'/TerrainSandbox/Ksn_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Ksn_Map')
            plt.ioff()
            fig = plt.figure(6)
            #imshow_grid(mg, "channel__steepness_index", grid_units=("m", "m"), var_name="Steepness Index", cmap="jet")
            Ksn = np.ones(np.size(mg.nodes)) * mg.at_node["channel__steepness_index"]
            Ksn[np.where(mg.node_x >= ((dxy * number_of_columns) - (dxy * 2)))] = 0
            Ksn[np.where(mg.node_y >= ((dxy * number_of_rows) - (dxy * 2)))] = 0
            Ksn[np.where(mg.node_x <= dxy * 2)] = 0
            Ksn[np.where(mg.node_y <= dxy * 2)] = 0
            imshow_grid(mg, Ksn, grid_units=("m", "m"), var_name="Steepness Index", cmap="jet")
            title_text = '$Year$='+str(total_time_r)  
            plt.title(title_text)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Ksn_Map/'+str(total_time_r)+'.'+Export_format,  format=Export_format,  dpi=dpi)
            plt.close(fig)
        
        # Chi_Profile
        if Chi_Profile == True:
            prf.run_one_step()
            cf.calculate_chi()
            if path.exists(str(Directory)+'/TerrainSandbox/Chi_Profile') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Chi_Profile')
            plt.ioff()
            fig = plt.figure(7)
            for i, outlet_id in enumerate(prf.data_structure):
                for j, segment_id in enumerate(prf.data_structure[outlet_id]):
                    if j == 0:
                        label = "channel {i}".format(i=i + 1)
                    else:
                        label = '_nolegend_'
                    segment = prf.data_structure[outlet_id][segment_id]
                    profile_ids = segment["ids"]
                    color = segment["color"]
                    plt.plot(mg.at_node["channel__chi_index"][profile_ids], mg.at_node["topographic__elevation"][profile_ids], color=color, label=label)
            plt.xlabel("Chi Index (m)")
            plt.ylabel("Elevation (m)")
            plt.legend(loc="lower right")
            title_text = '$Year$='+str(total_time_r)  
            plt.title(title_text)
            plt.grid(linestyle='--')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Chi_Profile/'+str(total_time_r)+'.'+Export_format,  format=Export_format,  dpi=dpi)
            plt.close(fig)
        
        # Chi_Map
        if Chi_Map == True:
            cf.calculate_chi()
            if path.exists(str(Directory)+'/TerrainSandbox/Chi_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Chi_Map')
            plt.ioff()
            fig = plt.figure(8)
            imshow_grid(mg, "channel__chi_index", grid_units=("m", "m"), var_name="Chi Index", cmap="jet")
            title_text = '$Year$='+str(total_time_r)  
            plt.title(title_text)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Chi_Map/'+str(total_time_r)+'.'+Export_format,  format=Export_format,  dpi=dpi)
            plt.close(fig)
        
        # Erosion_Rate_Profile
        if Erosion_Rate_Profile == True:
            prf.run_one_step()
            if path.exists(str(Directory)+'/TerrainSandbox/Erosion_Rate_Profile') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Erosion_Rate_Profile')
            plt.ioff()
            fig = plt.figure(3)
            prf.plot_profiles(field=erosionrate, xlabel='Distance Upstream (m)', ylabel='Erosion Rate (m/yr)')
            title_text = '$Year$='+str(total_time_r)
            plt.title(title_text)
            plt.grid(linestyle='--')
            axes = plt.gca()
            axes.set_xlim([dxy,None])
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Erosion_Rate_Profile/'+str(total_time_r)+'.'+Export_format,  format=Export_format, dpi=dpi)
            plt.close(fig)
        
        # Erosion_Rate_Map
        if Erosion_Rate_Map == True:
            if path.exists(str(Directory)+'/TerrainSandbox/Erosion_Rate_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Erosion_Rate_Map')
            plt.ioff()
            fig = plt.figure(8)
            eet.map_erosionrate()
            title_text = '$Year$='+str(total_time_r)  
            plt.title(title_text)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Erosion_Rate_Map/'+str(total_time_r)+'.'+Export_format, format=Export_format, dpi=dpi)
            plt.close(fig)
        
        # DZ_DT_Profile
        if DZ_DT_Profile == True:
            prf.run_one_step()
            if path.exists(str(Directory)+'/TerrainSandbox/DZDT_Profile') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/DZDT_Profile')
            plt.ioff()
            fig = plt.figure(3)
            prf.plot_profiles(field=dzdt, xlabel='Distance Upstream (m)', ylabel='Rate of Elevational Change (m/yr)')
            title_text = '$Year$='+str(total_time_r)
            plt.title(title_text)
            plt.grid(linestyle='--')
            axes = plt.gca()
            axes.set_xlim([dxy,None])
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/DZDT_Profile/'+str(total_time_r)+'.'+Export_format, format=Export_format, dpi=dpi)
            plt.close(fig)
        
        # DZ_DT_Map
        if DZ_DT_Map == True:
            if path.exists(str(Directory)+'/TerrainSandbox/DZDT_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/DZDT_Map')
            plt.ioff()
            fig = plt.figure(8)
            eet.map_dzdt()
            title_text = '$Year$='+str(total_time_r)  
            plt.title(title_text)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/DZDT_Map/'+str(total_time_r)+'.'+Export_format, format=Export_format, dpi=dpi)
            plt.close(fig)
            
        # Mean_Basin_Erosion_Rate
        if Mean_Basin_Erosion_Rate == True:
            if path.exists(str(Directory)+'/TerrainSandbox/Mean_Basin_Erosion_Rate') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Mean_Basin_Erosion_Rate')
            plt.ioff()
            fig = plt.figure()
            #x_axis = np.arange(dt, total_time + dt, dt)
            plt.plot(times, mean_basin_erosion_rate, '.--')
            title_text = '$Year$='+str(total_time_r) 
            plt.title(title_text)
            plt.grid(linestyle='--')
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Mean_Basin_Erosion_Rate/Mean_Basin_Erosion_Rate.' + Export_format,  format = Export_format, dpi=dpi)
            plt.close(fig)  
            
        # Net_Erosion_Map
        if Net_Erosion_Map == True:
            if path.exists(str(Directory)+'/TerrainSandbox/Net_Erosion_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Net_Erosion_Map')
            plt.ioff()
            fig = plt.figure()
            net_erosion = (U * total_time) - (mg.at_node['topographic__elevation'] - mg0.at_node['topographic__elevation'])
            imshow_grid(mg, net_erosion , grid_units=("m", "m"), var_name="Net erosion", cmap="jet")
            title_text = '$Year$='+str(total_time)  
            plt.title(title_text)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Net_Erosion_Map/'+str(total_time_r)+'.'+Export_format, format=Export_format, dpi=dpi)
            plt.close(fig)
            
        # Net_DZ_Map
        if Net_DZ_Map == True:
            if path.exists(str(Directory)+'/TerrainSandbox/Net_DZ_Map') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Net_DZ_Map')
            plt.ioff()
            fig = plt.figure()
            net_dz = (mg.at_node['topographic__elevation'] - mg0.at_node['topographic__elevation'])
            imshow_grid(mg, net_dz, grid_units=("m", "m"), var_name="Net dz", cmap="jet")
            title_text = '$Year$='+str(total_time_r)  
            plt.title(title_text)
            plt.tight_layout()
            fig.savefig(str(Directory)+'/TerrainSandbox/Net_DZ_Map/'+str(total_time)+'.'+Export_format, format=Export_format, dpi=dpi)
            plt.close(fig)
            
        # Reset Plot_Ticker    
        Plot_Ticker = 0
        
    # Export_DEM    
    if Export_DEM == True:
        if total_time_r == Export_DEM_Ticker:
            if path.exists(str(Directory)+'/TerrainSandbox') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox')
        if Export_DEM_Ticker >= Export_DEM_Interval - 0.0000000001:
            if path.exists(str(Directory)+'/TerrainSandbox/Export_DEM') == False:
                os.mkdir(str(Directory)+'/TerrainSandbox/Export_DEM')
            write_esri_ascii(str(Directory)+'/TerrainSandbox/Export_DEM/'+str(total_time_r)+'.asc', mg, names='topographic__elevation')
            Export_DEM_Ticker = 0
            
print('')
print('Complete!')





#%%

np.savetxt(Directory + 'TerrainSandbox/mean_basin_erosion_rate.csv', mean_basin_erosion_rate, delimiter = ",")
np.savetxt(Directory + 'TerrainSandbox/sum_basin_erosion.csv', sum_basin_erosion, delimiter = ",")
np.savetxt(Directory + 'TerrainSandbox/times.csv', times, delimiter = ",")
np.savetxt(Directory + 'TerrainSandbox/timesteps.csv', timesteps, delimiter = ",")
#
np.savetxt(Directory + 'TerrainSandbox/std_basin_erosion_rate.csv', std_basin_erosion_rate, delimiter = ",")
df = pd.DataFrame.from_dict(stats_list)
df.to_csv(Directory + 'TerrainSandbox/erosion_stats.csv', index = False, header=True)
  

  