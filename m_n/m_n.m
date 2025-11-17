%% Stream Power Exponent (m and n) Optimization Script
% Written by Chris Sheehan

%% Reset

% Reset
clear all
close all
clc

%% Directories

% Get master directory
master_directory = fileparts(pwd);

% Make output directory
mkdir('Output');

%% Parameters

% Grid parameters
dxy = [26.6736749983000]        % DEM resolution (m)
min_area = 2                    % Minimum drainage area to prune network (cells)    

% Export parameters
file_type = 'pdf'
file_res = 300

% Analysis paramters
i = 1000        % Iterations

%% Import DEM and get metrics

% Import DEM
DEMr = GRIDobj([master_directory '\Input\Chestatee_utm32616.tif']);

% Create FD, A, and S for each DEM
DEMr = fillsinks(DEMr)
FDr = FLOWobj(DEMr);
Ar = flowacc(FDr);
Sr = STREAMobj(FDr, Ar >= min_area);

%% Initialize arrays and figures

% Initialize
m = [];
n = [];
meanm = [];
meann = [];
stdm = [];
stdn = [];
dmeanm = NaN;
dmeann = NaN;
dstdm = NaN;
dstdn = NaN;
fig3 = figure(3);
fig4 = figure(4);
fig5 = figure(5);
fig6 = figure(6);
fig7 = figure(7);
fig8 = figure(8);

%% Iterate through mn optimization

% Iterate
for i = 1 : 1000
    
    % Display status
    disp(['Iteration ', num2str(i)])
    
    % Perform optimization
    [mn,results] = mnoptim(Sr, DEMr, Ar, 'lossfun', 'linear', 'crossval', false, 'optvar', 'm&n');
    
    % Store results
    m(i, 1) = mn.(1);
    n(i, 1) = mn.(2);
    meanm(i, 1) = mean(m);
    meann(i, 1) = mean(n);
    stdm(i, 1) = std(m);
    stdn(i, 1) = std(n);
    if i > 1
        dmeanm(i, 1) = abs(meanm(i, 1) - meanm(i - 1, 1));
        dmeann(i, 1) = abs(meann(i, 1) - meann(i - 1, 1));
        dstdm(i, 1) = abs(stdm(i, 1) - stdm(i - 1, 1));
        dstdn(i, 1) = abs(stdn(i, 1) - stdn(i - 1, 1));
    end

    % Close figs
    close fig 1;
    close fig 2;
    
    % Plot progress m
    figure(3);
    x = 1 : i;
    errorbar(x, meanm, stdm, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
    xlabel('iteration')
    ylabel('mean m')
    
    % Plot progress n
    figure(4);
    x = 1 : i;
    errorbar(x, meann, stdn, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
    xlabel('iteration')
    ylabel('mean n')
    
    % Plot progress dmeanm
    figure(5);
    x = 1 : i;
    semilogy(x, dmeanm, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
    xlabel('iteration')
    ylabel('d(mean m)')
    
    % Plot progress dmeann
    figure(6);
    x = 1 : i;
    semilogy(x, dmeann, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
    xlabel('iteration')
    ylabel('d(mean n)')
    
    % Plot progress dstdm
    figure(7);
    x = 1 : i;
    semilogy(x, dstdm, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
    xlabel('iteration')
    ylabel('d(std m)')
    
    % Plot progress dstdn
    figure(8);
    x = 1 : i;
    semilogy(x, dstdn, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
    xlabel('iteration')
    ylabel('d(std n)')         
 
end

%% Plot final figures

% Plot progress m
fig3 = figure(3);
x = 1 : i;
errorbar(x, meanm, stdm, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'LineWidth', 0.1);
xlabel('iteration')
ylabel('mean m')
exportgraphics(fig3, ['Output/meanm.' file_type], 'Resolution', file_res);
close fig 3;

% Plot progress n
fig4 = figure(4);
x = 1 : i;
errorbar(x, meann, stdn, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'LineWidth', 0.1);
xlabel('iteration')
ylabel('mean n')
exportgraphics(fig4, ['Output/meann.' file_type], 'Resolution', file_res);
close fig 4;

% Plot progress dmeanm
fig5 = figure(5);
x = 1 : i;
semilogy(x, dmeanm, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
xlabel('iteration')
ylabel('d(mean m)')
exportgraphics(fig5, ['Output/dmeanm.' file_type], 'Resolution', file_res);
close fig 5;

% Plot progress dmeann
fig6 = figure(6);
x = 1 : i;
semilogy(x, dmeann, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
xlabel('iteration')
ylabel('d(mean n)')
exportgraphics(fig6, ['Output/dmeann.' file_type], 'Resolution', file_res);
close fig 6;

% Plot progress dstdm
fig7 = figure(7);
x = 1 : i;
semilogy(x, dstdm, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
xlabel('iteration')
ylabel('d(std m)')
exportgraphics(fig7, ['Output/dstdm.' file_type], 'Resolution', file_res);
close fig 7;

% Plot progress dstdn
fig8 = figure(8);
x = 1 : i;
semilogy(x, dstdn, '-o', 'Color', 'k', 'MarkerSize', 5, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
xlabel('iteration')
ylabel('d(std n)')   
exportgraphics(fig8, ['Output/dstdn.' file_type], 'Resolution', file_res);
close fig 8;
