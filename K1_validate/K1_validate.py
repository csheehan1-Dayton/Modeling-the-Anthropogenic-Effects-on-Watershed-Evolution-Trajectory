#%% Script to validate K1 calculation with a rating curve
# **Author**: Chris Sheehan

#%% Import libraries

# Print status
print('Importing libraries...')

import inspect
import os
from matplotlib import pyplot as plt
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
from os import path
import os.path
import os
import pandas as pd
from numpy import nan, isnan
import numpy as np

# Section break
print(' ')

#%% Set directories

# Print status
print('Setting directories...')

# Set directories
script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_directory = script_directory.replace('\k1_validate', '')
export_directory = script_directory + '/'

# Create directories
if path.exists(str(export_directory)+'/K1_validate_output') == False:
    os.mkdir(str(export_directory)+'/K1_validate_output')

# Section break
print(' ')

#%% Parameters

# Print status
print('Setting parameters...')

# Stream gauge 02333500 (Chestateee River Near Dahlonega)
basin_area = 608568641.6459954          # m^2, entire Chestaee
gauge_upstream_area = 396268179.3       # m^2, upstream of gauge

# Suspended sediment data at gauge
Gauge_Sed_path = project_directory + r'\Input\02333500_Sed.csv'
Gauge_Hydro_path = project_directory + r'\Input\02333500_Hydro.csv'

# Export parameters
Export_format = 'png'
dpi = 600

# Set Euler's number
e = 2.71828

# Section break
print(' ')

#%% Import and clean data

# Print status
print('Importing and cleaning data...')

# Import and clean gauge sed data
skiprows = np.arange(0, 36)
skiprows = np.append(skiprows, 37)
Gauge_Sed = pd.read_csv(Gauge_Sed_path, skiprows=[1, 2])  # nrows = 545

# Convert short tons per day to cubic meters per year. 907.18474 kg per short ton, 2650 kg per cubic meter (quartz), 365 days per year
Gauge_Sed['sediment_flux_m3_year'] = Gauge_Sed['Suspended sediment discharge, short tons per day'] * (907.18474) * (1 / 2650) * (365)
    
# Convert cfs to m^3 / yr.
Gauge_Sed['water_discharge_m3_year'] = Gauge_Sed['Enforced Discharge, cfs'] * 893000.074

# Import and clean gauge hydro data
Gauge_Hydro = pd.read_csv(Gauge_Hydro_path, skiprows=[1, 2])  # nrows = 545
Gauge_Hydro['daily_discharge_m3yr-1'] = nan
Gauge_Hydro['daily_discharge_m3yr-1'] = Gauge_Hydro['36255_00060_00003'] * 893000.074          # cfs to m3yr-1

# Section break
print(' ')

#%% Create rating curve

# Print status
print('Creating rating curve...')

# Extract log values of gauge sed data
Qw = Gauge_Sed['water_discharge_m3_year'].values
Qw = Qw.astype(float)
Qw = np.log(Qw)
#
Qs = Gauge_Sed['sediment_flux_m3_year'].values
Qs = Qs.astype(float)
Qs = np.log(Qs)

# Remove nans
delete = np.ones(np.size(Qw)) * nan
for i in np.arange(0, np.size(Qw)):
    if isnan(Qw[i]) == True or isnan(Qs[i]) == True:
        delete[i] = True
    else:
        delete[i] = False
delete = np.where(delete == True)
delete = delete[0]
Qw = np.delete(Qw, delete)
Qs = np.delete(Qs, delete)

# Fit Regression.                           
polyfit = np.polyfit(Qw, Qs, deg=1)
m = polyfit[0]
b = polyfit[1]

# Create xaxis values for regression plot
line_x = np.array([np.min(Qw), np.max(Qw)])
line_y = (line_x * m) + b

# Section break
print(' ')

#%% Plot rating curves

# Print status
print('Plotting rating curves...')

# Plot
plt.ioff()
fig = plt.figure()
plt.scatter(Qw, Qs, 5, zorder=1, c=Gauge_Sed['Year'])
plt.plot(line_x, line_y)
plt.xlabel('log(water discharge) $(m^{{3}}$ $yr^{-1})$')
plt.ylabel('log(sediment discharge) $(m^{3}$ $yr^{-1})$')
plt.text(20.5, 4, 'y = ' + str(m) + 'x ' + str(b))
plt.grid()
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_validate_output/Rating_Curve.' + Export_format, dpi=dpi)
plt.close(fig)

# Use Rating Curve
plt.ioff()
fig = plt.figure()
plt.scatter(Qw, Qs, 5, zorder=1)
plt.plot(line_x, line_y, 'k--')
plt.xlabel('log(water discharge) $(m^{{3}}$ $yr^{-1})$')
plt.ylabel('log(sediment discharge) $(m^{3}$ $yr^{-1})$')
plt.grid()
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_validate_output/Rating_Curve_Use.' + Export_format, dpi=dpi)
plt.close(fig)

# Plot in groups
plt.ioff()
fig = plt.figure()
#
plt.scatter(Qw[0: 21], Qs[0: 21], 5, zorder=1, color='blue')
polyfit = np.polyfit(Qw[0: 21], Qs[0: 21], deg=1)
m = polyfit[0]
b = polyfit[1]
line_x = np.array([np.min(Qw[0: 21]), np.max(Qw[0: 21])])
line_y = (line_x * m) + b
plt.plot(line_x, line_y, color='blue')
#
plt.scatter(Qw[21: 59], Qs[21: 59], 5, zorder=1, color='green')
polyfit = np.polyfit(Qw[21: 59], Qs[21: 59], deg=1)
m = polyfit[0]
b = polyfit[1]
line_x = np.array([np.min(Qw[21: 59]), np.max(Qw[21: 59])])
line_y = (line_x * m) + b
plt.plot(line_x, line_y, color='green')
#
plt.scatter(Qw[59: 159], Qs[59: 159], 5, zorder=1, color='orange')
polyfit = np.polyfit(Qw[59: 159], Qs[59: 159], deg=1)
m = polyfit[0]
b = polyfit[1]
line_x = np.array([np.min(Qw[59: 159]), np.max(Qw[59: 159])])
line_y = (line_x * m) + b
plt.plot(line_x, line_y, color='orange')
#
plt.scatter(Qw[159: 273], Qs[159: 273], 5, zorder=1, color='red')
polyfit = np.polyfit(Qw[159: 273], Qs[159: 273], deg=1)
m = polyfit[0]
b = polyfit[1]
line_x = np.array([np.min(Qw[159: 273]), np.max(Qw[159: 273])])
line_y = (line_x * m) + b
plt.plot(line_x, line_y, color='red')
plt.legend(['1957 - 1963 (n = 21)', '', '1972 - 1976 (n = 38)', '', '1989 - 1994 (n = 100)', '', '1995 - 1998 (n = 114)', ''])
#
plt.xlabel('log(water discharge) $(m^{{3}}$ $yr^{-1})$')
plt.ylabel('log(sediment discharge) $(m^{3}$ $yr^{-1})$')
plt.text(20.5, 4, 'y = ' + str(m) + 'x ' + str(b))
plt.grid()
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_validate_output/Rating_Curve_Grouped.' + Export_format, dpi=dpi)
plt.close(fig)

# Section break
print(' ')

# %% Use rating curve to calculate annual sediment export

# Print status
print('Calculating annual sediment export...')

# Initiate variables
Qw = []
Qs = []
Qs_annual = []
Qs_max_annual = []
max_Qs = 0
Qs_sum = 0

# Loop through years
for i in np.arange(0, np.size(Gauge_Hydro['daily_discharge_m3yr-1'])):
    
    # Grab values for given year
    Qw = Gauge_Hydro['daily_discharge_m3yr-1'][i]
    Qw = np.log(Qw)
    Qs = (Qw * m) + b
    Qs = e ** Qs

    # Account for time
    Qs *= (1 / 365)     # dt = 1 / 365

    # Sum Qs
    Qs_sum += Qs
    
    # Update max Qs
    if Qs > max_Qs:
        max_Qs = Qs
        
    # Append
    if i < np.size(Gauge_Hydro['daily_discharge_m3yr-1']) - 1:
        if Gauge_Hydro['year'][i + 1] > Gauge_Hydro['year'][i]:
            Qs_annual = np.append(Qs_annual, Qs_sum)
            Qs_max_annual = np.append(Qs_max_annual, max_Qs)
            Qs_sum = 0
            max_Qs = 0
            
    # Append
    if i == np.size(Gauge_Hydro['daily_discharge_m3yr-1']) - 1:
        Qs_annual = np.append(Qs_annual, Qs_sum)
        Qs_max_annual = np.append(Qs_max_annual, max_Qs)
        Qs_sum = 0
        max_Qs = 0
       
# Calculate mean catchment erosion rate
mcer = Qs_annual / gauge_upstream_area
mcer_mean = np.nanmean(mcer)
mcer_std = np.nanstd(mcer)
mcer_25 = np.nanpercentile(mcer, 25)
mcer_75 = np.nanpercentile(mcer, 75)

# Section break
print(' ')

#%% Compute Values to Report

# Print status
print('Computing values to report...')

Qs_annual_mean = np.nanmean(Qs_annual)              # Mean annual sediment volume exported from watershed (m^3)
Qs_annual_min = np.nanquantile(Qs_annual, 0.25)     # Min annual sediment volume exported from watershed (m^3). Defined as 25th percentile, because std(Qs_annual) > mean(Qs_annual)
Qs_annual_max = np.nanquantile(Qs_annual, 0.75)     # Max annual sediment volume exported from watershed (m^3). Defined as 75th percentile, because std(Qs_annual) > mean(Qs_annual)
E = mcer_mean
E_min = np.nanquantile(mcer, 0.25) 
E_max = np.nanquantile(mcer, 0.75) 

# Section break
print(' ')

#%% Analysis plots

# Print status
print('Plotting analyses...')

# Estimated_Annual_Exported_Sediment
plt.ioff()
fig = plt.figure()
ax = fig.add_subplot()
xaxis = np.arange(1930, 2022)
xaxis = xaxis.astype(str)
plt.plot(xaxis, Qs_annual, 'ko')
xaxis[1: -1] = ''
ax.set_xticklabels(xaxis, rotation=45)
plt.xlabel('Year')
plt.ylabel('Estimated annual sediment flux $(m^{3})$')
plt.grid()
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_validate_output/Estimated_Annual_Exported_Sediment.' + Export_format, dpi=dpi)
plt.close(fig)

# Sediment Export
plt.ioff()
fig = plt.figure()
ax = fig.add_subplot()
xaxis = np.arange(1930, 2022)
xaxis = xaxis.astype(str)
plt.plot(xaxis, Qs_annual, 'ko', label='Estimated annual total')
plt.axhspan(Qs_annual_min, Qs_annual_max, alpha = 0.5, color = 'k', label='Interquartile range')
plt.axhline(Qs_annual_mean, color='r', linestyle='-', label='Mean')
xaxis[1: -1] = ''
ax.set_xticklabels(xaxis, rotation=45)
plt.yscale('log')
plt.ylim(1E2, 1E5)
plt.xlabel('Year')
plt.ylabel('Annual sediment export $(m^{3})$')
plt.grid()
plt.legend()
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_validate_output/Estimated_sediment_export.' + Export_format, dpi=dpi)
plt.close(fig)

# Mean catchment erosion rate
plt.ioff()
fig = plt.figure()
ax = fig.add_subplot()
xaxis = np.arange(1930, 2022)
xaxis = xaxis.astype(str)
plt.plot(xaxis, mcer, 'ko', label='Estimated annual rate')
plt.axhspan(E_min, E_max, alpha = 0.5, color = 'k', label='Interquartile range')
plt.axhline(E, color='g', linestyle='-', label='Mean')
plt.axhline(np.nanmedian(mcer), color='orange', linestyle='-', label='Median')
xaxis[1: -1] = ''
ax.set_xticklabels(xaxis, rotation=45)
plt.yscale('log')
plt.ylim(1E-6, 1E-3)
plt.xlabel('Year')
plt.ylabel('Mean catchment erosion rate $(m^{3} yr^{-1})$')
plt.grid()
plt.legend()
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_validate_output/Estimated_catchment_averaged_erosion_rates.' + Export_format, dpi=dpi)
plt.close(fig)

# Past and Present Mean catchment erosion rate
plt.ioff()
fig = plt.figure()
ax = fig.add_subplot()
xaxis = np.arange(1930, 2022)
xaxis = xaxis.astype(str)
plt.plot([], [], marker='None', linestyle='None', label =  r"$\bf{" + 'Rating' + "}$" + ' ' + r"$\bf{" + 'curve' + "}$")
plt.plot(xaxis, mcer, 'ko', label='Estimated annual rates')
plt.axhline(E, color='g', linestyle='-', label='Mean')
plt.axhline(np.nanmedian(mcer), color='orange', linestyle='-', label='Median')
plt.axhspan(E_min, E_max, alpha = 0.5, color = 'k', label='Interquartile range')
plt.plot([], [], marker='None', linestyle='None', label= r'$\bf{^{10}Be}$' )
plt.axhline(7.96E-06, color='b', linestyle='-', label='mean')
plt.axhspan(6.19E-06, 9.73E-06, alpha = 0.5, color = 'c', label='+/- 1 sigma')
plt.plot([], [], marker='None', linestyle='None', label=' ' )
xaxis[1: -1] = ''
ax.set_xticklabels(xaxis, rotation=45)
plt.yscale('log')
plt.ylim(1E-6, 1E-3)
plt.xlabel('Year')
plt.ylabel('Mean catchment erosion rate $(m^{3} yr^{-1})$')
plt.grid()
plt.legend(loc=2, ncol=2, prop={'size': 6})
plt.tight_layout()
fig.savefig(str(export_directory)+'/K1_validate_output/Both_Catchment_Averaged_Erosion_Rates.' + Export_format, dpi=dpi)
plt.close(fig)

# E1 vs E1rc
plt.ioff()
fig, ax = plt.subplots()
plt.boxplot(mcer[~np.isnan(mcer)], showmeans=True, meanline=True)
plt.errorbar(1.3, 2.94694647344916E-05, 6.55288348995580E-06, capsize=2, ecolor='red')
plt.ylabel('Mean catchment erosion rate $(m^{3} yr^{-1})$')
fig.set_figwidth(2)
plt.tight_layout()
ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.set_xticks([])
plt.ylim(1E-6, 1E-3)
plt.yscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
fig.savefig(str(export_directory)+'/K1_validate_output/E1_E1rc_comparison.' + Export_format, dpi=dpi)
plt.close(fig)

# E1 vs E1rc with 10Be
plt.ioff()
fig, ax = plt.subplots()
plt.boxplot(mcer[~np.isnan(mcer)], showmeans=True, meanline=True)
plt.errorbar(1.3, 2.94694647344916E-05, 6.55288348995580E-06, capsize=2, ecolor='red')
plt.errorbar(1.6, 7.96E-06, 1.77E-06, capsize=2, ecolor='b')
plt.errorbar(3, 7.96E-06, 1.77E-06, alpha=0)
plt.ylabel('Mean catchment erosion rate $(m^{3} yr^{-1})$')
fig.set_figwidth(3)
plt.tight_layout()
ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.set_xticks([])
plt.ylim(1E-6, 1E-3)
plt.yscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
fig.savefig(str(export_directory)+'/K1_validate_output/E1_E1rc_10Be_comparison.' + Export_format, dpi=dpi)
plt.close(fig)

# Mean catchment erosion rate
plt.ioff()
#
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#
xaxis = np.arange(1930, 2022)
xaxis = xaxis.astype(str)
ax1.plot(xaxis, mcer, 'ko', label='Estimated annual rate')
ax1.axhspan(E_min, E_max, alpha = 0.25, color = 'k', label='Interquartile range')
ax1.axhline(E, color='g', linestyle='-', label='Mean')
ax1.axhline(np.nanmedian(mcer), color='orange', linestyle='-', label='Median')
xaxis[1: -1] = ''
ax1.set_xticklabels(xaxis, rotation=45)
plt.yscale('log')
plt.ylim(1E-6, 1E-3)
ax1.set_xlabel('Year')
ax1.set_ylabel('Mean catchment erosion rate $(m^{3} yr^{-1})$')
ax1.grid(which='both', axis='y', linestyle='--', linewidth=0.5)
ax1.legend(prop={'size': 6})
plt.tight_layout()
#
bp1 = ax2.boxplot(mcer[~np.isnan(mcer)], showmeans=True, meanline=True, patch_artist=True, boxprops=dict(facecolor="k", alpha = 0.25))
eb1 = ax2.errorbar(1.3, 2.94694647344916E-05, 6.55288348995580E-06, capsize=2, color='red', ecolor='red', label='Test2')
eb2 = ax2.errorbar(1.6, 7.96E-06, 1.77E-06, capsize=2, color='blue', ecolor='b', label='Test3')
ax2.xaxis.set_tick_params(labelbottom=False)
ax2.grid(which='both', axis='y', linestyle='--', linewidth=0.5)
ax2.legend([bp1["boxes"][0], eb1, eb2], ['Rating curve data', '$E_{1}$', '$^{10}Be$'], loc='upper right', prop={'size': 6})
#
fig.savefig(str(export_directory)+'/K1_validate_output/Combined.' + Export_format, dpi=dpi)
plt.close(fig)

# Section break
print(' ')

#%% Print values

# Print status
print('Displaying values...')

print('Mean annual sediment flux: ', np.nanmean(Qs_annual), ' m^3 yr^-1')
print('Median annual sediment flux: ', np.nanmedian(Qs_annual), ' m^3 yr^-1')
print('Min annual sediment flux: ', np.nanmin(Qs_annual), ' m^3 yr^-1')
print('Max annual sediment flux: ', np.nanmax(Qs_annual), ' m^3 yr^-1')
print('2-siga lower annual sediment flux: ', np.nanpercentile(Qs_annual, 2.3), ' m^3 yr^-1')
print('2-siga higher annual annual sediment flux: ', np.nanpercentile(Qs_annual, 97.7), ' m^3 yr^-1')
print('25th percentile annual sediment flux: ', np.nanpercentile(Qs_annual, 25), ' m^3 yr^-1')
print('75th percentile annual annual sediment flux: ', np.nanpercentile(Qs_annual, 75), ' m^3 yr^-1')

# Section break
print(' ')

#%% Finalize

# Print status
print('Finished!')