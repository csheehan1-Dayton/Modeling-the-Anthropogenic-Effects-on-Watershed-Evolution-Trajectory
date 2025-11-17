%% D* Optimization Script
% Written by Chris Sheehan

%% Reset

% Reset
clear all
close all
clc

%% Directories

% Print status
disp('Setting directories...')

% Get master directory
master_directory = fileparts(pwd);

% Set input directory
direct = 'Create_Terrains_to_Verify_D_star_output/Final_DEMs/';

% Get list of grids in 'Create_Terrains_to_Verify_D_prime_output/Final_DEMs/'
contents = dir(direct);
names = {contents.name};
list = {};
i = 1;
for c = 1 : size(names, 2)
    if contains(names{c}, '.tif') == 1 
        if contains(names{c}, '.aux') == 0 
            list{i} = names{c};
            i = i + 1;
        end
    end
end

% Convert list strings to numbers
list_convert = list;
list_num = [];
for i = 1 : size(list, 2);
    list_convert{i} = erase(list_convert{i}, '.tif');
    list_convert{i} = str2double(list_convert{i});
    list_num(1, i) = list_convert{i};
end

% Make output directory
mkdir('D_star_Calib_output');

% Section break
disp(' ')

%% Parameters

% Print status
disp('Setting directories...')

% Grid parameters
dxy = [26.6736749983000]    % DEM resolution (m)
min_area = 1                % Minimum drainage area to prune network (m^2)

% Export parameters
file_type = 'png'
file_res = 600

% Section break
disp(' ')

%% Analyze real DEM

% Print status
disp('Analyzing real DEM...')

% Inport Chestatee DEM
[DEMc,FDc,Ac,Sc] = MakeStreams([master_directory '\Input\Chestatee_utm32616.tif'], min_area, 'no_data_exp','auto');

% Based on inspection of Chestatee DEM
min_slope = 0.04;
min_grad = (dxy^2) * 2;

% Slope-area of Chestatee DEM
fig = figure(1);
hold on;
SAc = slopearea(Sc, DEMc, Ac);
title('Real DEM');
grid();
exportgraphics(fig, ['D_star_Calib_output/SA_Chestatee.' file_type], 'Resolution', file_res);
close fig 1;

% Plot Chestatee DEM
fig = figure(1);
imageschs(DEMc, [], 'colormap', 'turbo');
title('Real DEM')
exportgraphics(fig, ['D_star_Calib_output/DEM_Chestatee.' file_type], 'Resolution', file_res);
close fig 1;

% Find values to use in regression
use1 = SAc.g > min_slope;
use2 = SAc.a > min_grad;
use = use1 & use2;

% Regress slope-area
xc = log(SAc.a(use));
yc = log(SAc.g(use));
pc = polyfit(xc, yc, 1);

% Section break
disp(' ')

%% Analyze model grids

% Print status
disp('Looping through model grids...')

% Sort list_num
[sorted, indices] = sort(list_num);

% Initiate variables
Rm_mean = [];
Rm_std = [];
maxg_index = [];

% Loop through grids
for i = 1 : size(list, 2)
    
    % Makestreams
    DEMi = GRIDobj([direct list{i}]);
    
    % Handle no-data values
    for r = 1 : size(DEMi.Z, 1)
        for c = 1 : size(DEMi.Z, 2)
            if DEMi.Z(r,c) == -99999
                DEMi.Z(r,c) = NaN;
            end
        end
    end
    
    % Create FD, A, and S for each DEM
    DEMi = fillsinks(DEMi)
    FDm = FLOWobj(DEMi);
    Am = flowacc(FDm);
    Sm = STREAMobj(FDm, Am >= (min_area / (dxy^2))); 
    
    % Slope-area relationship of model
    fig = figure(1);
    SAi = slopearea(Sm, DEMi, Am);
    title(['Model Grid (D* = ' erase(list{i}, '.tif') ')']);
    grid();
    exportgraphics(fig, ['D_star_Calib_output/SA_' erase(list{i}, '.tif') '.' file_type], 'Resolution', file_res)
    close fig 1;
    
    % Regress slope-area
    xi = log(SAi.a(use));
    yi = log(SAi.g(use));
    pi = polyfit(xi, yi, 1);
    
    % Plot model DEM
    fig = figure(1);
    imageschs(DEMi, [], 'colormap', 'turbo');
    title(['Model Grid (D* = ' erase(list{i}, '.tif') ')'])
    exportgraphics(fig, ['D_star_Calib_output/DEM_' erase(list{i}, '.tif') '.' file_type], 'Resolution', file_res);
    close fig 1;  
    
    % Plot combined DEMs
    fig = figure(1);
    hold on;
    subplot(1, 2, 1)
    imageschs(DEMc, [], 'colormap', 'turbo');
    xlabel('real');
    subplot(1, 2, 2)
    hold on;
    imageschs(DEMi, [], 'colormap', 'turbo');
    xlabel('model');
    exportgraphics(fig, ['D_star_Calib_output/DEM_Comparison_' erase(list{i}, '.tif') '.' file_type], 'Resolution', file_res);
    close fig 1;  
    
    % Plot combined SAs
    fig = figure();
    hold on;
    plot(log(SAc.a), log(SAc.g), 'ks');
    plot(log(SAc.a), (log(SAc.a) * pc(1)) + pc(2), 'k');
    plot(log(SAi.a), log(SAi.g), 'bs');
    plot(log(SAi.a), (log(SAi.a) * pi(1)) + pi(2), 'b');
    title(['Model Grid (D* = ' erase(list{i}, '.tif') ')']);
    grid()
    xlabel('Drainage area (m^2)')
    ylabel('Slope')
    legend('Real', '', ['D* = ' erase(list{i}, '.tif')], '');
    exportgraphics(fig, ['D_star_Calib_output/SA_Comparison_' erase(list{i}, '.tif') '.' file_type], 'Resolution', file_res);
    close fig 1;  

    % Plot combined SAs, no trendline
    fig = figure();
    hold on;
    plot(log(SAc.a), log(SAc.g), 'ks');
    plot(log(SAi.a), log(SAi.g), 'bs');
    title(['Model Grid (D* = ' erase(list{i}, '.tif') ')']);
    grid()
    xlabel('Drainage area (m^2)')
    ylabel('Slope')
    legend('Real', ['D* = ' erase(list{i}, '.tif')]);
    exportgraphics(fig, ['D_star_Calib_output/SA_Comparison_clean_' erase(list{i}, '.tif') '.' file_type], 'Resolution', file_res);
    close fig 1;  
    
    % Find index of max g in SA object
    maxg_index(1, i) = find(SAi.g == max(SAi.g));
    if SAi.g(1) > SAi.g(3);
        j = 1;
    elseif SAi.g(1) < SAi.g(3);
        j = 3;
    end
    one_or_three_larger_slope(1, i) = j;

    % Store data to create figure for paper
    if string(list{i}) == '200.tif'
        X_low = SAi.a;
        Y_low = SAi.g;
    end
    %
    if string(list{i}) == '325.tif'
        X_high = SAi.a;
        Y_high = SAi.g;
    end

end

% Section break
disp(' ')

%% Summary Figure

% Print status
disp('Creating summary figure...')

% Create Figure for paper
fig = figure()
hold on;
plot(SAc.a, SAc.g, 'ks');
plot(X_low, Y_low, 'bs');
plot(X_high, Y_high, 'rs');
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
xlabel('Drainage area (m^2)');
ylabel('Slope');
grid();
legend('Real', 'D* = 200', 'D* = 325');
exportgraphics(fig, ['D_star_Calib_output/For_Paper.' file_type], 'Resolution', file_res);
close fig 1; 

% Section break
disp(' ')

%% Finalize

 % Print status
disp('Finished!')