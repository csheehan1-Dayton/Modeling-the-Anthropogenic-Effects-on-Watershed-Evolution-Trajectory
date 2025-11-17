#%% Script to empirically calculate K0 (Equation 5)
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
from landlab.io.esri_ascii import read_esri_ascii 
from landlab.components import FlowAccumulator 

# Section break
print(' ')

#%% Set directories

# Print status
print('Setting directories...')

# Set directories
script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_directory = script_directory.replace('\k0_empirical', '')
export_directory = script_directory + '/'

# Create directories
print('Creating directory...')
if path.exists(str(export_directory)+'/K0_empirical_output') == False:
    os.mkdir(str(export_directory)+'/K0_empirical_output')

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

# 10Be mean catchment erosion rate, assumed to be uplift rate
SAP56_mean_recalc = 7.96 * 1E-6         # m / yr^-1
SAP56_mean_recalc_std = 1.77 * 1E-6     # m / yr^-1

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
no_data_nodes = np.where(mg.at_node['topographic__elevation'] == no_data_value)
no_data_nodes = no_data_nodes[0]

# Handle Grid Boundaries
mg.set_status_at_node_on_edges(right=4, top=4, left=4, bottom=4)
mg.status_at_node[pour_point_node] = mg.BC_NODE_IS_FIXED_VALUE

# Section break
print(' ')

#%% Initialize components

# Print status
print('Initializing components...')

# Initialize FlowAccumulator
frr = FlowAccumulator(mg, flow_director='D8')
frr.run_one_step()

# Section break
print(' ')

#%% Calculate K0

# Print status
print('Calculating K0...')

# Mean catchment erosion rate from 10Be
E = SAP56_mean_recalc

# Number of grid nodes (accounting for no data nodes)
num_nodes = np.size(zr) - np.size(no_data_nodes)

# Calculate
Ai_m = mg.at_node['drainage_area'] ** m_sp
Ai_m[no_data_nodes] = nan
Si_n = mg.at_node['topographic__steepest_slope'] ** n_sp
Si_n[no_data_nodes] = nan
mult = Ai_m * Si_n
add = np.nansum(mult)
divd = add / num_nodes
K_empir = E / divd
K_empir_min = (E - SAP56_mean_recalc_std) / divd
K_empir_max = (E + SAP56_mean_recalc_std) / divd

# Place K0s in dataframe and export to csv
df = pd.DataFrame([{'K0': K_empir, 'K0l': K_empir_min, 'K0h': K_empir_max}])
df.to_csv('K0_empirical_output/K0_values.csv', index=False)

# Section break
print(' ')

#%% Finalize

# Print status
print('Finished!')