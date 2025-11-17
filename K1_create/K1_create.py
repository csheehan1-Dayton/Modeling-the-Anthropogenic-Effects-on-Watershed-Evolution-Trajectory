#%% Script to empirically calculate K1 (Equation 8) and create the K1 array
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
from matplotlib import pyplot as plt
from landlab.io.esri_ascii import read_esri_ascii, write_esri_ascii
from landlab.components import FlowAccumulator 
from landlab import imshow_grid

# Section break
print(' ')

#%% Set directories

# Print status
print('Setting directories...')

# Set directories
script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_directory = script_directory.replace('\k1_create', '')
export_directory = script_directory + '/'

# Create directories
print('Creating directory...')
if path.exists(str(export_directory)+'/K1_create_output') == False:
    os.mkdir(str(export_directory)+'/K1_create_output')

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

# Land cover parameters
lc_path = project_directory + r'\Input\NCLD_2019_Align.asc'
lc_scheme_path = project_directory + '\Input\Erosion_Rate_Scheme.csv'

# Export parameters
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
no_data_nodes = np.where(mg.at_node['topographic__elevation'] == no_data_value)
no_data_nodes = no_data_nodes[0]

# Handle Grid Boundaries
mg.set_status_at_node_on_edges(right=4, top=4, left=4, bottom=4)
mg.status_at_node[pour_point_node] = mg.BC_NODE_IS_FIXED_VALUE

# Import land cover data and erosion rate scheme
lc = read_esri_ascii(lc_path)
lc = lc[1]
lc_scheme = pd.read_csv(lc_scheme_path)

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

#%% Calculate E1 and create K1 array

# Print status
print('Calculating E1 and creating K1 array...')

# K0 values (copied from earlier work)
K0 = 8.29124436168032E-07
K0l = 6.44758826618105E-07
K0h = 1.01E-06

# Add erodibility grid node atttributes 
mg.add_ones('node', 'K1')
mg.add_ones('node', 'K1l')
mg.add_ones('node', 'K1h')
mg.at_node['K1'] *= nan
mg.at_node['K1l'] *= nan
mg.at_node['K1h'] *= nan

# Loop through grid nodes
for i in np.arange(0, np.size(lc)):
    
    # Print status
    if i % max(1, np.size(lc) // 100) == 0:
        print(f"{(i / np.size(lc)) * 100:.0f}% complete")
    
    # Operate on non-NaN nodes
    if zr[i] != no_data_value:
        
        # Get land cover id at node
        value = lc[i]
        
        # Get cooresponding erodibility value
        r = np.where(lc_scheme['nlcd_value'] == value)
        r = r[0]
        
        # Set erodibility value at node
        mg.at_node['K1'][i] = lc_scheme['Normalized Prefered C Factor (Ci)'][r] * K0
        mg.at_node['K1l'][i] = lc_scheme['Normalized Prefered C Factor (Ci)'][r] * K0l
        mg.at_node['K1h'][i] = lc_scheme['Normalized Prefered C Factor (Ci)'][r] * K0h
        
# Export 1D csv of all values in K1 array   s     
np.savetxt(str(export_directory)+'/K1_create_output/K1.csv', mg.at_node['K1'], delimiter = ",")
np.savetxt(str(export_directory)+'/K1_create_output/K1l.csv', mg.at_node['K1l'], delimiter = ",")
np.savetxt(str(export_directory)+'/K1_create_output/K1h.csv', mg.at_node['K1h'], delimiter = ",")

# Export asc grid of K1 array
mg.at_node['K1'][np.isnan(mg.at_node['K1'])] = -99999
mg.at_node['K1l'][np.isnan(mg.at_node['K1l'])] = -99999
mg.at_node['K1h'][np.isnan(mg.at_node['K1h'])] = -99999
write_esri_ascii(str(export_directory)+'/K1_create_output/K1.asc', mg, names = 'K1', clobber = True)
write_esri_ascii(str(export_directory)+'/K1_create_output/K1l.asc', mg, names = 'K1l', clobber = True)
write_esri_ascii(str(export_directory)+'/K1_create_output/K1h.asc', mg, names = 'K1h', clobber = True)

# Section break
print(' ')

#%% Calculate E1

# Print status
print('Calculating E1...')

# Convert no data values to NaN
mg.at_node['K1'][no_data_nodes] = nan
mg.at_node['K1l'][no_data_nodes] = nan
mg.at_node['K1h'][no_data_nodes] = nan

# Number of grid nodes (accounting for no data nodes)
num_nodes = np.size(zr) - np.size(no_data_nodes)                        

# Calculate
Ai_m = mg.at_node['drainage_area'] ** m_sp
Ai_m[no_data_nodes] = nan
Si_n = mg.at_node['topographic__steepest_slope'] ** n_sp
Si_n[no_data_nodes] = nan
E1 = np.nanmean(Ai_m * Si_n * mg.at_node['K1'])
E1l = np.nanmean(Ai_m * Si_n * mg.at_node['K1l'])
E1h = np.nanmean(Ai_m * Si_n * mg.at_node['K1h'])

# Place E1s in dataframe and export to csv
df = pd.DataFrame([{'E1': E1, 'E1l': E1l, 'E1h': E1h}])
df.to_csv('K1_create_output/E1_values.csv', index=False)

# Section break
print(' ')

#%% Plots

# Print status
print('Plotting figures...')

# Export K1 map
plt.ioff()
fig = plt.figure(figsize = (5, 3.5))
imshow_grid(mg, mg.at_node['K1'], var_name="$\mathregular{K_{sp}}$", allow_colorbar=True)
plt.tick_params(left = False, bottom=False) 
plt.xticks([])  
plt.yticks([]) 
plt.xlabel('') 
plt.ylabel('') 
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_create_output/K1.png', format='png', dpi = dpi)
fig.savefig(str(export_directory)+'/K1_create_output/K1.pdf', format='pdf', dpi = dpi)
plt.close(fig)

# Export focus area 1
plt.ioff()
fig = plt.figure(figsize = (5, 3.5))
imshow_grid(mg, mg.at_node['K1'], var_name="$\mathregular{K_{sp}}$", allow_colorbar=True)
plt.tick_params(left = False, bottom=False) 
plt.xticks([])  
plt.yticks([]) 
plt.xlabel('') 
plt.ylabel('') 
plt.xlim([776395, 781890]) 
plt.ylim([3820650, 3826330]) 
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_create_output/F1.png', format='png', dpi = dpi)
fig.savefig(str(export_directory)+'/K1_create_output/F1.pdf', format='pdf', dpi = dpi)
plt.close(fig)

# Export focus area 2
plt.ioff()
fig = plt.figure(figsize = (5, 3.5))
imshow_grid(mg, mg.at_node['K1'], var_name="$\mathregular{K_{sp}}$", allow_colorbar=True)
plt.tick_params(left = False, bottom=False) 
plt.xticks([])  
plt.yticks([]) 
plt.xlabel('') 
plt.ylabel('') 
plt.xlim([775050, 783490])  
plt.ylim([3839835, 3848560]) 
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_create_output/F2.png', format='png', dpi = dpi)
fig.savefig(str(export_directory)+'/K1_create_output/F2.pdf', format='pdf', dpi = dpi)
plt.close(fig)

# Export focus area 3
plt.ioff()
fig = plt.figure(figsize = (5, 3.5))
imshow_grid(mg, mg.at_node['K1'], var_name="$\mathregular{K_{sp}}$", allow_colorbar=True)
plt.tick_params(left = False, bottom=False) 
plt.xticks([])  
plt.yticks([]) 
plt.xlabel('') 
plt.ylabel('') 
plt.xlim([789481, 797220])     
plt.ylim([3823370, 3831370])  
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_create_output/F3.png', format='png', dpi = dpi)
fig.savefig(str(export_directory)+'/K1_create_output/F3.pdf', format='pdf', dpi = dpi)
plt.close(fig)

# Section break
print(' ')

#%% Finalize

# Print status
print('Finished!')