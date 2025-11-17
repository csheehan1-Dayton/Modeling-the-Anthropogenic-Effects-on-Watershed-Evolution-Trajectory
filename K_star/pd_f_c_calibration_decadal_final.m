%% pd, f, and c Calculation  Script
% Written by Chris Sheehan

%% Reset

% Reset
clear all
close all
clc

%% Directories

% Print status
disp('Setting directories...')

% Get directories
master_directory = fileparts(pwd);

% Make output directory
mkdir('pd_f_c_calibration_decadal_output');

% Toggle off figure visibility
set(0, 'DefaultFigureVisible', 'off');

% Section break
disp(' ')

%% Parameters

% Print status
disp('Setting parameters...')

% Precipitation parameters
dp = 0.001;
pmax = 1000;
Im = 15;
m = 0.503;

% Export parameters
file_type = 'png'
file_res = 600

% Section break
disp(' ')

%% Map coordinates

% Print status
disp('Mapping coordinates...')

% Import DEM
DEM = GRIDobj([master_directory '\Input\Chestatee_utm32616.tif']);

% Import Coords
Coords = readmatrix('Coords.csv');

% Create directory
mkdir('pd_f_c_calibration_decadal_output/Maps');

% Plot
fig = figure(1);
hold on;
imageschs(DEM, [], 'colormap', 'turbo');
plot(Coords(:, 7), Coords(:, 6), 'ko', 'MarkerFaceColor', 'r');
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/Maps/MACA_Input_Coordinate_Map.', file_type], 'Resolution', file_res)
close fig 1
%
fig = figure(1);
hold on;
imageschs(DEM, [], 'colormap', 'turbo');
plot(Coords(:, 8), Coords(:, 9), 'ks', 'MarkerFaceColor', 'y');
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/Maps/MACA_Cell_Center_Map.', file_type], 'Resolution', file_res)
close fig 1
%
fig = figure(1);
hold on;
imageschs(DEM, [], 'colormap', 'turbo');
plot(Coords(:, 7), Coords(:, 6), 'ko', 'MarkerFaceColor', 'r');
plot(Coords(:, 8), Coords(:, 9), 'ks', 'MarkerFaceColor', 'y');
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/Maps/MACA_Combined_Map.', file_type], 'Resolution', file_res)
close fig 1

% Section break
disp(' ')

%% Historical data

% Print status
disp('Parsing and plotting historical data...')

% Create "List" of csv files
folder = [master_directory '/Input/MACA_Data/Linear/Hist'];
contents = dir(folder);
names = {contents.name};
List = {};
i = 1;
for c = 1 : size(names, 2)
    if contains(names{c}, '.csv') == 1 
        List{i} = names{c};
        i = i + 1;       
    end
end
%
clear c;
clear contents;
clear i;
clear names;

% Initiate "C"
Pd = (1 : size(List, 2)) * NaN;
F = (1 : size(List, 2)) * NaN;
C = (1 : size(List, 2)) * NaN;

% Create Output_pd_f_c_calibration_linear folder
mkdir('pd_f_c_calibration_decadal_output/Hist');

% Add csv files to "p"
for i = 1 : size(List, 2)
    
    % Create Output_pd_f_c_calibration_linear folder
    mkdir(['pd_f_c_calibration_decadal_output/Hist/' num2str(i)])

    % Import file
    p = readmatrix([folder '/' num2str(i) '.csv']);
    
    % Calculate record length (days)
    record_length = size(p, 1);
    
    % Delete time column
    p(:, 1) = [];
    
    % Remove nans and zeros
    p(isnan(p)) = [];       % Remove nans
    p(p == 0) = [];         % Remove zeros

    % Calculate "pd" (method of doing this partially inside loop above and in 
    % this line is for consistancy with Python script that calculates pd and F).
    pd = mean(p);
    Pd(i) = pd;
    
    % Calculate f
    f = size(p, 1) / record_length;
    F(i) = f;
    
    % Find unique values of p
    p_unique = unique(p);

    % Calculate number of events larger than each p_unique
    count = p_unique * nan;
    for j = 1 : size(p_unique, 1)
        count(j) = sum(p > p_unique(j));  
    end

    % Calculate exceedence frequency
    Pr = count / size(p, 1);

    % Plot Pr
    fig1 = figure(1);
    semilogy(p_unique, Pr, '.');
    ax = gca;
    color = ax.ColorOrder(ax.ColorOrderIndex - 1, :);
    
    % Calculate double ln transformed data
    y = log(Pr);
    y = y * -1;           
    y = log(y);
    %
    x = log(p_unique);
    y(end) = [];
    x(end) = [];
    
    % ployfit
    pf = polyfit(x(x>0), y(x>0), 1);
    c = pf(1);
    
    % Store "c"
    C(i) = c;
    
    % Plot log-transformed data and export
    fig2 = figure(2);
    hold on;
    plot(x, y, '.');
    plot(x, (x*c) + pf(2));
    xlabel('ln(p)');
    ylabel('ln( ln(Pr) )');
    legend('data', 'fit');
    txt = ['c = ', num2str(c)];
    text(4, -6, txt);
    grid();
    exportgraphics(fig2, ['pd_f_c_calibration_decadal_output/Hist/' num2str(i) '/Log_Transformed.' file_type], 'Resolution', file_res)
    close fig 2;
    
    % Add extrapolation to Figure 1 and export
    figure(1);
    hold on;
    lamda = pd / (gamma(1 + (1 / c)));
    fit = exp( -(p_unique / lamda) .^ c );
    plot(p_unique, fit, 'Color', color);
    xlabel('p');
    ylabel('Pr');
    legend('data', 'fit');
    txt = ['c = ', num2str(c)];
    text(20, 10^-2, txt);
    grid();
    exportgraphics(fig1, ['pd_f_c_calibration_decadal_output/Hist/' num2str(i) '/p_vs_ExhedanceFrequancy.' file_type], 'Resolution', file_res)
    close fig 1;
    
end

% Store data
C_Hist_mean = mean(C);
C_Hist_std = std(C);
Pd_Hist_mean = mean(Pd);
Pd_Hist_std = std(Pd);
F_Hist_mean = mean(F);
F_Hist_std = std(F);

% Section break
disp(' ')

%% RCP 4.5

% Print status
disp('Parsing and plotting RCP 4.5 data...')

% Create "List" of csv files
folder = [master_directory '/Input/MACA_Data/Decadal/Combined/RCP45'];
contents = dir(folder);
names = {contents.name};
List = {};
i = 1;
for c = 1 : size(names, 2)
    if contains(names{c}, '.csv') == 1 
        List{i} = names{c};
        i = i + 1;       
    end
end
%
clear c;
clear contents;
clear i;
clear names;

% Initiate "C"
Pd = (1 : size(List, 2)) * NaN;
F = (1 : size(List, 2)) * NaN;
C = (1 : size(List, 2)) * NaN;

% Create output folder
mkdir('pd_f_c_calibration_decadal_output/RCP45');

% Add csv files to "p"
for i = 1 : size(List, 2)
    
    % Create output folder
    mkdir(['pd_f_c_calibration_decadal_output/RCP45/' num2str(i)])

    % Import file
    p = readtable([folder '/' num2str(i) '.csv']);
    p.Properties.VariableNames = ["Date", "Prcp"];
    
    % Extract years
    Years = year(p.Date);
    
    % Initiate starting year index
    year_lower = 2000;
    year_upper = 2010;
    
    % Set indeces for loop below
    toggle = 1;
    row = 1;
    
    % Loop through decades
    while toggle == 1
        
        % Create directory for current decade
        mkdir(['pd_f_c_calibration_decadal_output/RCP45/' num2str(i) '/' num2str(year_lower)]);
        
        % Identify days within current decade
        use = Years >= year_lower & Years < year_upper;
        
        % Create subset of "p" only containing relevant decade
        subset = p(use, :);
        
        % Calculate record length (days)
        record_length = size(subset, 1);
        
        % Copy rainfall to separate array
        %subset(:, 1) = [];
        prcp = subset.Prcp;
        
        % Remove nans and zeros
        prcp(isnan(prcp)) = [];       % Remove nans
        prcp(prcp == 0) = [];         % Remove zeros
        
        % Calculate "pd" (method of doing this partially inside loop above and in 
        % this line is for consistancy with Python script that calculates pd and F).
        pd = mean(prcp);
        Pd(row, i) = pd;
        
        % Calculate f
        f = size(prcp, 1) / record_length;
        F(row, i) = f;
        
        % Find unique values of p
        prcp_unique = unique(prcp);
        
        % Calculate number of events larger than each p_unique
        count = prcp_unique * nan;
        for j = 1 : size(prcp_unique, 1)
            count(j) = sum(subset.Prcp > prcp_unique(j));  
        end
        
        % Calculate exceedence frequency
        Pr = count / size(prcp, 1);
        
        % Plot Pr
        fig1 = figure(1);
        semilogy(prcp_unique, Pr, '.');
        ax = gca;
    
        % Calculate double ln transformed data
        y = log(Pr);
        y = y * -1;           
        y = log(y);
        %
        x = log(prcp_unique);
        y(end) = [];
        x(end) = [];
        
        % Polyfit   
        pf = polyfit(x(x>0), y(x>0), 1);
        c = pf(1);
        
        % Store "c"
        C(row, i) = c;
        
        % Plot log-transformed data and export
        fig2 = figure(2);
        hold on;
        plot(x, y, '.');
        plot(x, (x*c) + pf(2));
        xlabel('ln(p)');
        ylabel('ln( ln(Pr) )');
        legend('data', 'fit');
        txt = ['c = ', num2str(c)];
        text(4, -6, txt);
        grid();
        exportgraphics(fig2, ['pd_f_c_calibration_decadal_output/RCP45/' num2str(i) '/' num2str(year_lower) '/Log_Transformed.' file_type], 'Resolution', file_res)
        close fig 2;
        
        % Add extrapolation to Figure 1 and export
        figure(1);
        hold on;
        lamda = pd / (gamma(1 + (1 / c)));
        fit = exp( -(prcp_unique / lamda) .^ c );
        plot(prcp_unique, fit);
        xlabel('p');
        ylabel('Pr');
        legend('data', 'fit');
        txt = ['c = ', num2str(c)];
        text(20, 10^-2, txt);
        grid();
        exportgraphics(fig1, ['pd_f_c_calibration_decadal_output/RCP45/' num2str(i) '/' num2str(year_lower) '/p_vs_ExhedanceFrequancy.' file_type], 'Resolution', file_res)
        close fig 1;
        
        % Advance indeces
        year_lower = year_lower + 10;
        year_upper = year_upper + 10;
        row = row + 1;
        
        if year_upper == 2110
            toggle = 0;
        end
  
    end
    
end

% Store data
C_RCP45_mean = NaN;
C_RCP45_std = NaN;
Pd_RCP45_mean = NaN;
Pd_RCP45_std = NaN;
F_RCP45_mean = NaN;
F_RCP45_std = NaN;
%
for r = 1 : size(Pd, 1)
    C_RCP45_mean(r, 1) = mean(C(r, :));
    C_RCP45_std(r,1) = std(C(r, :));
    Pd_RCP45_mean(r, 1) = mean(Pd(r, :));
    Pd_RCP45_std(r, 1) = std(Pd(r, :));
    F_RCP45_mean(r, 1) = mean(F(r, :));
    F_RCP45_std(r, 1) = std(F(r, :));
end

% Plot Pd
fig = figure();
x = 2000 : 10 : 2090;
errorbar(x, Pd_RCP45_mean(:, 1), Pd_RCP45_std(:, 1), '.-');
xlabel('Decade');
ylabel('Pd');
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/RCP45/Pd.'  file_type], 'Resolution', file_res);
close fig 1;

% Plot F
fig = figure();
x = 2000 : 10 : 2090;
errorbar(x, F_RCP45_mean(:, 1), F_RCP45_std(:, 1), '.-');
xlabel('Decade');
ylabel('F');
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/RCP45/F.'  file_type], 'Resolution', file_res);
close fig 1;

% Plot F
fig = figure();
x = 2000 : 10 : 2090;
errorbar(x, C_RCP45_mean(:, 1), C_RCP45_std(:, 1), '.-');
xlabel('Decade');
ylabel('C');
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/RCP45/C.'  file_type], 'Resolution', file_res);
close fig 1;

% Section break
disp(' ')

%% RCP 8.5

% Print status
disp('Parsing and plotting RCP 8.5 data...')

% Create "List" of csv files
folder = [master_directory '/Input/MACA_Data/Decadal/Combined/RCP85'];
contents = dir(folder);
names = {contents.name};
List = {};
i = 1;
for c = 1 : size(names, 2)
    if contains(names{c}, '.csv') == 1 
        List{i} = names{c};
        i = i + 1;       
    end
end
%
clear c;
clear contents;
clear i;
clear names;

% Initiate "C"
Pd = (1 : size(List, 2)) * NaN;
F = (1 : size(List, 2)) * NaN;
C = (1 : size(List, 2)) * NaN;

% Create output folder
mkdir('pd_f_c_calibration_decadal_output/RCP85');

% Add csv files to "p"
for i = 1 : size(List, 2)
    
    % Create output folder
    mkdir(['pd_f_c_calibration_decadal_output/RCP85/' num2str(i)])

    % Import file
    p = readtable([folder '/' num2str(i) '.csv']);
    p.Properties.VariableNames = ["Date", "Prcp"];
    
    % Extract years
    Years = year(p.Date);
    
    % Initiate starting year index
    year_lower = 2000;
    year_upper = 2010;
    
    % Set indeces for loop below
    toggle = 1;
    row = 1;
    
    % Loop through decades
    while toggle == 1
        
        % Create directory for current decade
        mkdir(['pd_f_c_calibration_decadal_output/RCP85/' num2str(i) '/' num2str(year_lower)]);
        
        % Identify days within current decade
        use = Years >= year_lower & Years < year_upper;
        
        % Create subset of "p" only containing relevant decade
        subset = p(use, :);
        
        % Calculate record length (days)
        record_length = size(subset, 1);
        
        % Copy rainfall to separate array
        prcp = subset.Prcp;
        
        % Remove nans and zeros
        prcp(isnan(prcp)) = [];       % Remove nans
        prcp(prcp == 0) = [];         % Remove zeros
        
        % Calculate "pd" (method of doing this partially inside loop above and in 
        % this line is for consistancy with Python script that calculates pd and F).
        pd = mean(prcp);
        Pd(row, i) = pd;
        
        % Calculate f
        f = size(prcp, 1) / record_length;
        F(row, i) = f;
        
        % Find unique values of p
        prcp_unique = unique(prcp);
        
        % Calculate number of events larger than each p_unique
        count = prcp_unique * nan;
        for j = 1 : size(prcp_unique, 1)
            count(j) = sum(subset.Prcp > prcp_unique(j));  
        end
        
        % Calculate exceedence frequency
        Pr = count / size(prcp, 1);
        
        % Plot Pr
        fig1 = figure(1);
        semilogy(prcp_unique, Pr, '.');
        ax = gca;
    
        % Calculate double ln transformed data
        y = log(Pr);
        y = y * -1;           
        y = log(y);
        %
        x = log(prcp_unique);
        y(end) = [];
        x(end) = [];
        
        % Polyfit    
        pf = polyfit(x(x>0), y(x>0), 1);
        c = pf(1);
        
        % Store "c"
        C(row, i) = c;
        
        % Plot log-transformed data and export
        fig2 = figure(2);
        hold on;
        plot(x, y, '.');
        plot(x, (x*c) + pf(2));
        xlabel('ln(p)');
        ylabel('ln( ln(Pr) )');
        legend('data', 'fit');
        txt = ['c = ', num2str(c)];
        text(4, -6, txt);
        grid();
        exportgraphics(fig2, ['pd_f_c_calibration_decadal_output/RCP85/' num2str(i) '/' num2str(year_lower) '/Log_Transformed.' file_type], 'Resolution', file_res)
        close fig 2;
        
        % Add extrapolation to Figure 1 and export
        figure(1);
        hold on;
        lamda = pd / (gamma(1 + (1 / c)));
        fit = exp( -(prcp_unique / lamda) .^ c );
        plot(prcp_unique, fit);
        xlabel('p');
        ylabel('Pr');
        legend('data', 'fit');
        txt = ['c = ', num2str(c)];
        text(20, 10^-2, txt);
        grid();
        exportgraphics(fig1, ['pd_f_c_calibration_decadal_output/RCP85/' num2str(i) '/' num2str(year_lower) '/p_vs_ExhedanceFrequancy.' file_type], 'Resolution', file_res)
        close fig 1;
        
        % Advance indeces
        year_lower = year_lower + 10;
        year_upper = year_upper + 10;
        row = row + 1;
        
        if year_upper == 2110
            toggle = 0;
        end
  
    end
    
end

% Store data
C_RCP85_mean = NaN;
C_RCP85_std = NaN;
Pd_RCP85_mean = NaN;
Pd_RCP85_std = NaN;
F_RCP85_mean = NaN;
F_RCP85_std = NaN;
%
for r = 1 : size(Pd, 1)
    C_RCP85_mean(r, 1) = mean(C(r, :));
    C_RCP85_std(r,1) = std(C(r, :));
    Pd_RCP85_mean(r, 1) = mean(Pd(r, :));
    Pd_RCP85_std(r, 1) = std(Pd(r, :));
    F_RCP85_mean(r, 1) = mean(F(r, :));
    F_RCP85_std(r, 1) = std(F(r, :));
end

% Plot Pd
fig = figure();
x = 2000 : 10 : 2090;
errorbar(x, Pd_RCP85_mean(:, 1), Pd_RCP85_std(:, 1), '.-');
xlabel('Decade');
ylabel('Pd');
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/RCP85/Pd.'  file_type], 'Resolution', file_res);
close fig 1;

% Plot F
fig = figure();
x = 2000 : 10 : 2090;
errorbar(x, F_RCP85_mean(:, 1), F_RCP85_std(:, 1), '.-');
xlabel('Decade');
ylabel('F');
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/RCP85/F.'  file_type], 'Resolution', file_res);
close fig 1;

% Plot F
fig = figure();
x = 2000 : 10 : 2090;
errorbar(x, C_RCP85_mean(:, 1), C_RCP85_std(:, 1), '.-');
xlabel('Decade');
ylabel('C');
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/RCP85/C.'  file_type], 'Resolution', file_res);
close fig 1;

% Section break
disp(' ')

%% Calculate RCP45_Ratio

% Print status
disp('Calculating RCP 4.5 ratio...')

% Initiate
RCP45_Ratio = NaN;

% Loop
for r = 1 : size(Pd_RCP45_mean, 1)

    % Assign variables
    pd0 = Pd_Hist_mean;
    pd1 = Pd_RCP45_mean(r);
    F0 = F_Hist_mean;
    F1 = F_RCP45_mean(r);
    c0 = C_Hist_mean;
    c1 = C_RCP45_mean(r);

    % Set p array
    p = 0.01 : dp : pmax;

    % Calculate lamdas
    lamda0 = pd0 / gamma(1 + (1 / c0));
    lamda1 = pd1 / gamma(1 + (1 / c1));

    % Set index
    index = 1;

    % Integrate
    for j = Im : dp : pmax

        % Calc numerator and denominator
        Den(index) = ((j - Im)^m) * ( (c0 / lamda0) .* ( (j / lamda0) .^ (c0 - 1) ) .* ( exp( -(j / lamda0) .^ c0 ) ) ) * dp;
        Num(index) = ((j - Im)^m) * ( (c1 / lamda1) .* ( (j / lamda1) .^ (c1 - 1) ) .* ( exp( -(j / lamda1) .^ c1 ) ) ) * dp;

        % Advance index
        index = index + 1;

    end

    % Fractionalize
    Den = Den * F0;
    Num = Num * F1;

    %Integrate
    Num_sum = nansum(Num);
    Den_sum = nansum(Den);

    % Divide
    RCP45_Ratio(r, 1) = Num_sum / Den_sum;

end

% Plot K ratio
x = 2000 : 10 : 2090;
fig = figure();
hold on;
plot(x, RCP45_Ratio, 'o-');
xlabel('Decade');
ylabel('K Ratio');
title('RCP 4.5');
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/RCP45.'  file_type], 'Resolution', file_res);
close fig 1;

% Section break
disp(' ')

%% Calculate RCP85_Ratio

% Print status
disp('Calculating RCP 8.5 ratio...')

% Initiate
RCP85_Ratio = NaN;

% Loop
for r = 1 : size(Pd_RCP85_mean, 1)

    % Assign variables
    pd0 = Pd_Hist_mean;
    pd1 = Pd_RCP85_mean(r);
    F0 = F_Hist_mean;
    F1 = F_RCP85_mean(r);
    c0 = C_Hist_mean;
    c1 = C_RCP85_mean(r);

    % Set p array
    p = 0.01 : dp : pmax;

    % Calculate lamdas
    lamda0 = pd0 / gamma(1 + (1 / c0));
    lamda1 = pd1 / gamma(1 + (1 / c1));

    % Set index
    index = 1;

    % Integrate
    for j = Im : dp : pmax

        % Calc numerator and denominator
        Den(index) = ((j - Im)^m) * ( (c0 / lamda0) .* ( (j / lamda0) .^ (c0 - 1) ) .* ( exp( -(j / lamda0) .^ c0 ) ) ) * dp;
        Num(index) = ((j - Im)^m) * ( (c1 / lamda1) .* ( (j / lamda1) .^ (c1 - 1) ) .* ( exp( -(j / lamda1) .^ c1 ) ) ) * dp;

        % Advance index
        index = index + 1;

    end

    % Fractionalize
    Den = Den * F0;
    Num = Num * F1;

    % Integrate
    Num_sum = nansum(Num);
    Den_sum = nansum(Den);

    % Divide
    RCP85_Ratio(r, 1) = Num_sum / Den_sum;

end

% Plot K ratio
x = 2000 : 10 : 2090;
fig = figure();
hold on;
plot(x, RCP85_Ratio, 'o-');
xlabel('Decade');
ylabel('K Ratio');
title('RCP 8.5');
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/RCP85.'  file_type], 'Resolution', file_res);
close fig 1;

% Section break
disp(' ')

%% Plot combined and write csv

% Print status
disp('Plotting combined figures and writing CSV...')

% Plot combined K ratio
x = 2000 : 10 : 2090;
fig = figure();
hold on;
plot(x, RCP45_Ratio, 'o-');
plot(x, RCP85_Ratio, 'o-');
xlabel('Decade');
ylabel('K Ratio');
title('RCP 8.5');
legend('RCP 4.5', 'RCP 8.5')
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/K_Ratio_Combined.'  file_type], 'Resolution', file_res);
close fig 1;

% Table
T45 = table(RCP45_Ratio);
T85 = table(RCP85_Ratio);
writetable(T45, 'pd_f_c_calibration_decadal_output/RCP45_Data.csv');
writetable(T85, 'pd_f_c_calibration_decadal_output/RCP85_Data.csv');

% Section break
disp(' ')

%% Calculate RCP45_Ratio lower and upper limit

% Print status
disp('Calculating RCP 4.5 lower and upper limites...')

% Initiate
RCP45_Ratio_low = NaN;

% Loop
for r = 1 : size(Pd_RCP45_mean, 1)

    % Assign variables
    pd0 = Pd_Hist_mean;
    pd1 = Pd_RCP45_mean(r) - Pd_RCP45_std(r);
    F0 = F_Hist_mean;
    F1 = F_RCP45_mean(r) - F_RCP45_std(r);
    c0 = C_Hist_mean;
    c1 = C_RCP45_mean(r) + C_RCP45_std(r);

    % Set p array
    p = 0.01 : dp : pmax;

    % Calculate lamdas
    lamda0 = pd0 / gamma(1 + (1 / c0));
    lamda1 = pd1 / gamma(1 + (1 / c1));

    % Set index
    index = 1;

    % Integrate
    for j = Im : dp : pmax

        % Calc numerator and denominator
        Den(index) = ((j - Im)^m) * ( (c0 / lamda0) .* ( (j / lamda0) .^ (c0 - 1) ) .* ( exp( -(j / lamda0) .^ c0 ) ) ) * dp;
        Num(index) = ((j - Im)^m) * ( (c1 / lamda1) .* ( (j / lamda1) .^ (c1 - 1) ) .* ( exp( -(j / lamda1) .^ c1 ) ) ) * dp;

        % Advance index
        index = index + 1;

    end

    % Fractionalize
    Den = Den * F0;
    Num = Num * F1;

    %Integrate
    Num_sum = nansum(Num);
    Den_sum = nansum(Den);

    % Divide
    RCP45_Ratio_low(r, 1) = Num_sum / Den_sum;

end

% Initiate
RCP45_Ratio_high = NaN;

% Loop
for r = 1 : size(Pd_RCP45_mean, 1)

    % Assign variables
    pd0 = Pd_Hist_mean;
    pd1 = Pd_RCP45_mean(r) + Pd_RCP45_std(r);
    F0 = F_Hist_mean;
    F1 = F_RCP45_mean(r) + F_RCP45_std(r);
    c0 = C_Hist_mean;
    c1 = C_RCP45_mean(r) - C_RCP45_std(r);

    % Set p array
    p = 0.01 : dp : pmax;

    % Calculate lamdas
    lamda0 = pd0 / gamma(1 + (1 / c0));
    lamda1 = pd1 / gamma(1 + (1 / c1));

    % Set index
    index = 1;

    % Integrate
    for j = Im : dp : pmax

        % Calc numerator and denominator
        Den(index) = ((j - Im)^m) * ( (c0 / lamda0) .* ( (j / lamda0) .^ (c0 - 1) ) .* ( exp( -(j / lamda0) .^ c0 ) ) ) * dp;
        Num(index) = ((j - Im)^m) * ( (c1 / lamda1) .* ( (j / lamda1) .^ (c1 - 1) ) .* ( exp( -(j / lamda1) .^ c1 ) ) ) * dp;

        % Advance index
        index = index + 1;

    end

    % Fractionalize
    Den = Den * F0;
    Num = Num * F1;

    %Integrate
    Num_sum = nansum(Num);
    Den_sum = nansum(Den);

    % Divide
    RCP45_Ratio_high(r, 1) = Num_sum / Den_sum;

end

% Plot K ratio w/ stds
x = 2000 : 10 : 2090;
neg = RCP45_Ratio - RCP45_Ratio_low;
pos = RCP45_Ratio_high - RCP45_Ratio;
fig = figure();
hold on;
errorbar(x, RCP45_Ratio, neg, pos, 'o-');
xlabel('Decade');
ylabel('K Ratio');
title('RCP 4.5');
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/K_Ratio_45_errorbar.'  file_type], 'Resolution', file_res);
close fig 1;

% Section break
disp(' ')

%% Calculate RCP85_Ratio lower and upper limit

% Print status
disp('Calculating RCP 8.5 lower and upper limites...')

% Initiate
RCP85_Ratio_low = NaN;

% Loop
for r = 1 : size(Pd_RCP85_mean, 1)

    % Assign variables
    pd0 = Pd_Hist_mean;
    pd1 = Pd_RCP85_mean(r) - Pd_RCP85_std(r);
    F0 = F_Hist_mean;
    F1 = F_RCP85_mean(r) - F_RCP85_std(r);
    c0 = C_Hist_mean;
    c1 = C_RCP85_mean(r) + C_RCP85_std(r);

    % Set p array
    p = 0.01 : dp : pmax;

    % Calculate lamdas
    lamda0 = pd0 / gamma(1 + (1 / c0));
    lamda1 = pd1 / gamma(1 + (1 / c1));

    % Set index
    index = 1;

    % Integrate
    for j = Im : dp : pmax

        % Calc numerator and denominator
        Den(index) = ((j - Im)^m) * ( (c0 / lamda0) .* ( (j / lamda0) .^ (c0 - 1) ) .* ( exp( -(j / lamda0) .^ c0 ) ) ) * dp;
        Num(index) = ((j - Im)^m) * ( (c1 / lamda1) .* ( (j / lamda1) .^ (c1 - 1) ) .* ( exp( -(j / lamda1) .^ c1 ) ) ) * dp;

        % Advance index
        index = index + 1;

    end

    % Fractionalize
    Den = Den * F0;
    Num = Num * F1;

    %Integrate
    Num_sum = nansum(Num);
    Den_sum = nansum(Den);

    % Divide
    RCP85_Ratio_low(r, 1) = Num_sum / Den_sum;

end

% Initiate
RCP85_Ratio_high = NaN;

% Loop
for r = 1 : size(Pd_RCP85_mean, 1)

    % Assign variables
    pd0 = Pd_Hist_mean;
    pd1 = Pd_RCP85_mean(r) + Pd_RCP85_std(r);
    F0 = F_Hist_mean;
    F1 = F_RCP85_mean(r) + F_RCP85_std(r);
    c0 = C_Hist_mean;
    c1 = C_RCP85_mean(r) - C_RCP85_std(r);

    % Set p array
    p = 0.01 : dp : pmax;

    % Calculate lamdas
    lamda0 = pd0 / gamma(1 + (1 / c0));
    lamda1 = pd1 / gamma(1 + (1 / c1));

    % Set index
    index = 1;

    % Integrate
    for j = Im : dp : pmax

        % Calc numerator and denominator
        Den(index) = ((j - Im)^m) * ( (c0 / lamda0) .* ( (j / lamda0) .^ (c0 - 1) ) .* ( exp( -(j / lamda0) .^ c0 ) ) ) * dp;
        Num(index) = ((j - Im)^m) * ( (c1 / lamda1) .* ( (j / lamda1) .^ (c1 - 1) ) .* ( exp( -(j / lamda1) .^ c1 ) ) ) * dp;

        % Advance index
        index = index + 1;

    end

    % Fractionalize
    Den = Den * F0;
    Num = Num * F1;

    %Integrate
    Num_sum = nansum(Num);
    Den_sum = nansum(Den);

    % Divide
    RCP85_Ratio_high(r, 1) = Num_sum / Den_sum;

end

% Plot K ratio w/ stds
x = 2000 : 10 : 2090;
neg = RCP85_Ratio - RCP85_Ratio_low;
pos = RCP85_Ratio_high - RCP85_Ratio;
fig = figure();
hold on;
errorbar(x, RCP85_Ratio, neg, pos, 'o-');
xlabel('Decade');
ylabel('K Ratio');
title('RCP 8.5');
grid()
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/K_Ratio_85_errorbar.'  file_type], 'Resolution', file_res);
close fig 1;

% Section break
disp(' ')

%% Plot combined errorbars

% Print status
disp('Plot combined errorbars...')

% Plot K ratio w/ stds
fig = figure();
hold on;
x = 2000 : 10 : 2090;
%
neg = RCP45_Ratio - RCP45_Ratio_low;
pos = RCP45_Ratio_high - RCP45_Ratio;
errorbar(x, RCP45_Ratio, neg, pos, 'o-');
%
neg = RCP85_Ratio - RCP85_Ratio_low;
pos = RCP85_Ratio_high - RCP85_Ratio;
errorbar(x, RCP85_Ratio, neg, pos, 'o-');
%
xlabel('Decade');
ylabel('K Ratio');
legend('RCP 4.5', 'RCP 8.5');
%
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/K_Ratio_combined_errorbar.'  file_type], 'Resolution', file_res);
close fig 1;

% Plot horizontal spans
fig = figure();
hold on;
%
xi = 2010 : 10 : 2100;
x(1) = 2000;
j = 2;
for i = 1 : size(xi, 2)
    x(j) = xi(i);
    x(j + 1) = xi(i);
    j = j + 2;
end
x(end) = [];
%
posi = transpose(RCP45_Ratio_high); 
negi = transpose(RCP45_Ratio_low);
pos = NaN;
neg = NaN;
j = 1;
%
for i = 1 : size(xi, 2)
    pos(j) = posi(i);
    pos(j + 1) = posi(i);
    neg(j) = negi(i);
    neg(j + 1) = negi(i);
    j = j + 2;
end
%
x2 = [x, fliplr(x)];
inBetween = [pos, fliplr(neg)];
fill(x2, inBetween, 'r', 'FaceAlpha', 0.5);
%
posi = transpose(RCP85_Ratio_high); 
negi = transpose(RCP85_Ratio_low);
pos = NaN;
neg = NaN;
j = 1;
%
for i = 1 : size(xi, 2)
    pos(j) = posi(i);
    pos(j + 1) = posi(i);
    neg(j) = negi(i);
    neg(j + 1) = negi(i);
    j = j + 2;
end
%
x2 = [x, fliplr(x)];
inBetween = [pos, fliplr(neg)];
fill(x2, inBetween, [.5 0 .5], 'FaceAlpha', 0.5);
%
meani = transpose(RCP45_Ratio); 
mean = NaN;
j = 1;
%
for i = 1 : size(xi, 2)
    mean(j) = meani(i);
    mean(j + 1) = meani(i);
    j = j + 2;
end
%
plot(x, mean, 'r');
%
meani = transpose(RCP85_Ratio); 
mean = NaN;
j = 1;
%
for i = 1 : size(xi, 2)
    mean(j) = meani(i);
    mean(j + 1) = meani(i);
    j = j + 2;
end
%
plot(x, mean, 'Color' ,[.5 0 .5]);
%
plot([2000, 2100], [1, 1], 'k--');
%
ylim([0.8 1.4]);
%
xlabel('Year');
ylabel('K^{*}');
legend('RCP 4.5', 'RCP 8.5', 'location', 'northwest');
%
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/K_Ratio_combined_span.'  file_type], 'Resolution', file_res);
close fig 1;

% Plot Linear Change Over Time
fig = figure();
hold on;
%
x = [2000, 2100];
x2 = [x, fliplr(x)];
%
neg45 = [1, 0.998061353];
pos45 = [1, 1.172407782];
inBetween45 = [pos45, fliplr(neg45)];
%
neg85 = [1, 1.168260104];
pos85 = [1, 1.364418207];
inBetween85 = [pos85, fliplr(neg85)];
%
fill(x2, inBetween45, 'r', 'FaceAlpha', 0.5);
fill(x2, inBetween85, [0.5 0 0.5], 'FaceAlpha', 0.5);
%
plot(x, [1, 1.084426617], 'r');
plot(x, [1, 1.26514041], 'Color', [.5 0 .5]);
plot(x, [1, 1], 'k--');
%
ylim([0.8 1.4]);
xlabel('Year');
ylabel('K Ratio');
legend('RCP 4.5', 'RCP 8.5', 'location', 'northwest');
%
grid();
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/K_Ratio_Linear.'  file_type], 'Resolution', file_res);
close fig 1;

% Section break
disp(' ')

%% Final Figure

% Print status
disp('Plotting final figures...')

% Plot
fig = figure()
%
% Plot horizontal spans
subplot(2, 2, 1);
hold on;
%
xi = 2010 : 10 : 2100;
x(1) = 2000;
j = 2;
for i = 1 : size(xi, 2)
    x(j) = xi(i);
    x(j + 1) = xi(i);
    j = j + 2;
end
x(end) = [];
%
posi = transpose(RCP45_Ratio_high); 
negi = transpose(RCP45_Ratio_low);
pos = NaN;
neg = NaN;
j = 1;
%
for i = 1 : size(xi, 2)
    pos(j) = posi(i);
    pos(j + 1) = posi(i);
    neg(j) = negi(i);
    neg(j + 1) = negi(i);
    j = j + 2;
end
%
x2 = [x, fliplr(x)];
inBetween = [pos, fliplr(neg)];
fill(x2, inBetween, 'r', 'FaceAlpha', 0.5);
%
posi = transpose(RCP85_Ratio_high); 
negi = transpose(RCP85_Ratio_low);
pos = NaN;
neg = NaN;
j = 1;
%
for i = 1 : size(xi, 2)
    pos(j) = posi(i);
    pos(j + 1) = posi(i);
    neg(j) = negi(i);
    neg(j + 1) = negi(i);
    j = j + 2;
end
%
x2 = [x, fliplr(x)];
inBetween = [pos, fliplr(neg)];
fill(x2, inBetween, [.5 0 .5], 'FaceAlpha', 0.5);
%
meani = transpose(RCP45_Ratio); 
mean = NaN;
j = 1;
%
for i = 1 : size(xi, 2)
    mean(j) = meani(i);
    mean(j + 1) = meani(i);
    j = j + 2;
end
%
plot(x, mean, 'r');
%
meani = transpose(RCP85_Ratio); 
mean = NaN;
j = 1;
%
for i = 1 : size(xi, 2)
    mean(j) = meani(i);
    mean(j + 1) = meani(i);
    j = j + 2;
end
%
plot(x, mean, 'Color' ,[.5 0 .5]);
%
plot([2000, 2100], [1, 1], 'k--');
%
ylim([0.8 1.4]);
grid();
%
xlabel('Year');
ylabel('K Ratio');
legend('RCP 4.5', 'RCP 8.5', 'location', 'northwest');
%
subplot(2, 2, 2);
hold on
x = 2000 : 10 : 2090;
errorbar(x, Pd_RCP45_mean(:, 1), Pd_RCP45_std(:, 1), 'r.-');
errorbar(x, Pd_RCP85_mean(:, 1), Pd_RCP85_std(:, 1), 'm.-');
grid();
xlabel('Decade');
ylabel('Pd');
%
subplot(2, 2, 3);
hold on
x = 2000 : 10 : 2090;
errorbar(x, C_RCP45_mean(:, 1), C_RCP45_std(:, 1), 'r.-');
errorbar(x, C_RCP85_mean(:, 1), C_RCP85_std(:, 1), 'm.-');
grid();
xlabel('Decade');
ylabel('c');
%
subplot(2, 2, 4);
hold on
x = 2000 : 10 : 2090;
errorbar(x, F_RCP45_mean(:, 1), F_RCP45_std(:, 1), 'r.-');
errorbar(x, F_RCP85_mean(:, 1), F_RCP85_std(:, 1), 'm.-');
grid();
xlabel('Decade');
ylabel('F');
%
set(gcf,'Position',[0 0 1000 600]);
exportgraphics(fig, ['pd_f_c_calibration_decadal_output/All_Factors.'  file_type], 'Resolution', file_res);
close fig 1;

% Section break
disp(' ')

%% More table exports

% Print status
disp('Exporting tables...')

% Table
T45h = table(RCP45_Ratio_high);
T45l = table(RCP45_Ratio_low);
T85h = table(RCP85_Ratio_high);
T85l = table(RCP85_Ratio_low);
%
TPd45_mean = table(Pd_RCP45_mean);
TPd45_std = table(Pd_RCP45_std);
TPd85_mean = table(Pd_RCP85_mean);
TPd85_std = table(Pd_RCP85_std);
%
TC45_mean = table(C_RCP45_mean);
TC45_std = table(C_RCP45_std);
TC85_mean = table(C_RCP85_mean);
TC85_std = table(C_RCP85_std);
%
TF45_mean = table(F_RCP45_mean);
TF45_std = table(F_RCP45_std);
TF85_mean = table(F_RCP85_mean);
TF85_std = table(F_RCP85_std);
%
writetable(T45h, 'pd_f_c_calibration_decadal_output/RCP45_High_Data.csv');
writetable(T45l, 'pd_f_c_calibration_decadal_output/RCP45_Low_Data.csv');
writetable(T85h, 'pd_f_c_calibration_decadal_output/RCP85_High_Data.csv');
writetable(T85l, 'pd_f_c_calibration_decadal_output/RCP85_Low_Data.csv');
%
writetable(TPd45_mean, 'pd_f_c_calibration_decadal_output/Pd45_mean_Data.csv');
writetable(TPd45_std, 'pd_f_c_calibration_decadal_output/Pd45_std.csv');
writetable(TPd85_mean, 'pd_f_c_calibration_decadal_output/Pd85_mean_Data.csv');
writetable(TPd85_std, 'pd_f_c_calibration_decadal_output/Pd85_std.csv');
%
writetable(TC45_mean, 'pd_f_c_calibration_decadal_output/C45_mean_Data.csv');
writetable(TC45_std, 'pd_f_c_calibration_decadal_output/C45_std.csv');
writetable(TC85_mean, 'pd_f_c_calibration_decadal_output/C85_mean_Data.csv');
writetable(TC85_std, 'pd_f_c_calibration_decadal_output/C85_std.csv');
%
writetable(TF45_mean, 'pd_f_c_calibration_decadal_output/F45_mean_Data.csv');
writetable(TF45_std, 'pd_f_c_calibration_decadal_output/F45_std.csv');
writetable(TF85_mean, 'pd_f_c_calibration_decadal_output/F85_mean_Data.csv');
writetable(TF85_std, 'pd_f_c_calibration_decadal_output/F85_std.csv');

% Section break
disp(' ')

%% Export K_star data

% Print status
disp('Exporting K_star data for models...')

% Create empty K_star arrays
K_star_45 = transpose(1 : 1000) * NaN;
K_star_45l = transpose(1 : 1000) * NaN;
K_star_45h = transpose(1 : 1000) * NaN;
K_star_85 = transpose(1 : 1000) * NaN;
K_star_85l = transpose(1 : 1000) * NaN;
K_star_85h = transpose(1 : 1000) * NaN;

% Fill arrays
for i = 0 : 9
    K_star_45((i * 100) + 1 : (i * 100) + 100) = T45{i + 1, 1};
    K_star_45l((i * 100) + 1 : (i * 100) + 100) = T45l{i + 1, 1};
    K_star_45h((i * 100) + 1 : (i * 100) + 100) = T45h{i + 1, 1};
    K_star_85((i * 100) + 1 : (i * 100) + 100) = T85{i + 1, 1};
    K_star_85l((i * 100) + 1 : (i * 100) + 100) = T85l{i + 1, 1};
    K_star_85h((i * 100) + 1 : (i * 100) + 100) = T85h{i + 1, 1};
end

% Export
writematrix(K_star_45, 'pd_f_c_calibration_decadal_output/K_star_45.csv');
writematrix(K_star_45l, 'pd_f_c_calibration_decadal_output/K_star_45l.csv');
writematrix(K_star_45h, 'pd_f_c_calibration_decadal_output/K_star_45h.csv');
writematrix(K_star_85, 'pd_f_c_calibration_decadal_output/K_star_85.csv');
writematrix(K_star_85l, 'pd_f_c_calibration_decadal_output/K_star_85l.csv');
writematrix(K_star_85h, 'pd_f_c_calibration_decadal_output/K_star_85h.csv');

% Section break
disp(' ')

%% Finalize

% Print status
disp('Finished!')