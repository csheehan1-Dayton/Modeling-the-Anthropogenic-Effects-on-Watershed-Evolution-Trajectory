%% K0 Validation  Script
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
direct = 'Create_Terrains_to_Validate_K0_output/Final_DEMs/';

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
mkdir('K0_validate_output');

% Section break
disp(' ')

%% Parameters

% Print status
disp('Setting parameters...')

% Grid parameters
min_area = 1                % Minimum drainage area to prune network (m^2)

% Stream power parameters
m_sp = 0.503
n_sp = 1.224

% Export parameters
file_type = 'png'
file_res = 600

% Section break
disp(' ')

%% Analyze real DEM

% Print status
disp('Analyzing real DEM...')

% Inport Chestatee DEM
[DEMr,FDr,Ar,Sr] = MakeStreams([master_directory '\Input\Chestatee_utm32616.tif'], min_area, 'no_data_exp','auto');

% Slope-area of real DEM
fig = figure(1);
hold on;
SAr = slopearea(Sr, DEMr, Ar);
title('Real DEM');
grid();
exportgraphics(fig, ['K0_validate_output/SA_real.' file_type], 'Resolution', file_res);
close fig 1;

% Logrithm of SAr data
alogr = log(SAr.a);
glogr = log(SAr.g);

% Polyfit data
fit = fitlm(alogr, glogr);

% Transpose DEMr
zlistr = [];
for c = 1 : size(DEMr.Z, 2)
    zlistr = [zlistr; DEMr.Z(:, c)];
end

% Plot real DEM
fig = figure(1);
imageschs(DEMr, [], 'colormap', 'turbo');
title('Real DEM')
c = colorbar;                    
ylabel(c, 'Elevation (m)');
exportgraphics(fig, ['K0_validate_output/DEM_real.' file_type], 'Resolution', file_res);
close fig 1;

% Gamma distribution of elevations
pd = fitdist(zlistr, 'gamma');
y = pdf(pd, zlistr);

% Find elevation of max y
max(y);
find(y == ans);
zlistr(ans);
maxpdfr = mean(ans);

% Plot single pdf
fig = figure(1);
hold on;
plot(zlistr, y, '.');
xlabel('Elevation (m)');
ylabel('pdf');
title('Real DEM');
grid();
exportgraphics(fig, ['K0_validate_output/pdf_real.' file_type], 'Resolution', file_res);
close fig 1;

% Plot combined pdf
fig = figure(3);
hold on;
plot(zlistr, y, 'K-', 'LineWidth', 1);

% Calculate real Ksn
ksnr = ksn(Sr, DEMr, Ar, m_sp / n_sp);
ksnr = smooth(Sr, ksnr);

% Map real Ksn values
fig = figure(1);
hold on;
imageschs(DEMr, [], 'colorbar', false, 'colormap', 'gray');
scatter(Sr.x, Sr.y, 5, ksnr, 'filled'); 
caxis([0 max(ksnr)]);
c = colorbar;                    
ylabel(c, 'K_{sn}');
title('Real DEM');
exportgraphics(fig, ['K0_validate_output/ksn_real.' file_type], 'Resolution', file_res);
close fig 1;

% Record values
mr = fit.Coefficients{2, 1};
br = fit.Coefficients{1, 1};
r2r = fit.Rsquared.Adjusted;
thetar = SAr.theta;
ksr = SAr.ks; 
zmeanr = nanmean(nanmean(DEMr.Z));
zstdr = nanstd(nanstd(DEMr.Z));
zmaxr = nanmax(nanmax(DEMr.Z));
zmedianr = nanmedian(nanmedian(DEMr.Z));

% Create empty arrays
m = ones(size(list, 2)) * NaN;
b = ones(size(list, 2)) * NaN;
r2 = ones(size(list, 2)) * NaN;
theta = ones(size(list, 2)) * NaN;
ks = ones(size(list, 2)) * NaN;
zmean = ones(size(list, 2)) * NaN;
zstd = ones(size(list, 2)) * NaN;
zmax = ones(size(list, 2)) * NaN;
zmedian = ones(size(list, 2)) * NaN;

% Plot combined SA
fig = figure(4);
hold on
p = polyfit(log10(SAr.a), log10(SAr.g), 1);
y = polyval(p, log10(SAr.a));
hold on;
plot(10.^(log10(SAr.a)), 10.^(y), 'DisplayName', ['Real'], 'LineWidth', 1, 'Color', 'k');
plot(10.^(log10(SAr.a)), 10.^(log10(SAr.g)), 's', 'Color', 'k');
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log')

% Boxplot
B(1:size(zlistr, 1), 1) = zlistr;

% Section break
disp(' ')

%% Analyze model grids

% Print status
disp('Looping through model grids...')

% Loop through grids
for i = 1 : size(list, 2)
    
    % Makestreams
    DEMm = GRIDobj([direct list{i}]);
    
    % Handle no-data values
    for r = 1 : size(DEMm.Z, 1)
        for c = 1 : size(DEMm.Z, 2)
            if DEMm.Z(r,c) == -99999
                DEMm.Z(r,c) = NaN;
            end
        end
    end
    
    % Create FD, A, and S for each DEM
    DEMm = fillsinks(DEMm)
    FDm = FLOWobj(DEMm);
    Am = flowacc(FDm);
    Sm = STREAMobj(FDm, Am >= min_area); 
    
    % Slope-area relationship of model
    fig = figure(1);
    SAm = slopearea(Sm, DEMm, Am);
    title(['Model Grid (K_{sp} = ' erase(list{i}, '.tif') ')']);
    grid();
    exportgraphics(fig, ['K0_validate_output/SA_' erase(list{i}, '.tif') '.' file_type], 'Resolution', file_res)
    close fig 1;
    
    % Logrithm of SAr data
    alogm = log(SAm.a);
    glogm = log(SAm.g);

    % Polyfit data
    fit = fitlm(alogm, glogm);
    
    % Record values
    m(1, i) = fit.Coefficients{2, 1};
    b(1, i) = fit.Coefficients{1, 1};
    r2(1, i) = fit.Rsquared.Adjusted;
    theta(1, i) = SAm.theta;
    ks(1, i) = SAm.ks;
    zmean(1, i) = nanmean(nanmean(DEMm.Z));
    zstd(1, i) = nanstd(nanstd(DEMm.Z));
    zmax(1, i) = nanmax(nanmax(DEMm.Z));
    zmedian(1, i) = nanmedian(nanmedian(DEMm.Z));
    
    % Transpose DEMm
    zlistm = [];
    for c = 1 : size(DEMm.Z, 2)
        zlistm = [zlistm; DEMm.Z(:, c)];
    end  
    
    % Plot model DEM
    fig = figure(1);
    imageschs(DEMm, [], 'colormap', 'turbo');
    c = colorbar;                    
    ylabel(c, 'Elevation (m)');
    title(['Model Grid (K_{sp} = ' erase(list{i}, '.tif') ')']);
    exportgraphics(fig, ['K0_validate_output/DEM_' erase(list{i}, '.tif') '.' file_type], 'Resolution', file_res);
    close fig 1;  
    
    % Plot model vs real DEM
    fig = figure(1);
    subplot(1, 2, 1);
    hold on;
    imageschs(DEMr, [], 'colormap', 'turbo');
    %caxis([min(zlistr) max(zlistr)]);
    c = colorbar;                    
    ylabel(c, 'Elevation (m)');
    xlabel('Real')
    subplot(1, 2, 2);
    hold on;
    imageschs(DEMm, [], 'colormap', 'turbo');
    %caxis([min(zlistr) max(zlistr)]);
    c = colorbar;                    
    ylabel(c, 'Elevation (m)');
    xlabel(['Model Grid (K_{sp} = ' erase(list{i}, '.tif') ')'])
    exportgraphics(fig, ['K0_validate_output/DEM_comparison_' erase(list{i}, '.tif') '.' file_type], 'Resolution', file_res);
    close fig 1;         
    
    % Gamma distribution
    pd = fitdist(zlistm, 'gamma');
    y = pdf(pd, zlistm);
    
    % Find elevation of max y
    max(y);
    find(y == ans);
    zlistm(ans);
    maxpdf(1, i) = mean(ans);
    
    % Plot single
    fig = figure(1);
    hold on;
    plot(zlistm, y, '.');
    xlabel('Elevation (m)');
    ylabel('pdf');
    title(['Model Grid (K_{sp} = ' erase(list{i}, '.tif') ')']);
    grid();
    exportgraphics(fig, ['K0_validate_output/pdf_' erase(list{i}, '.tif') '.' file_type], 'Resolution', file_res);
    close fig 1;

    % Plot Combined
    fig = figure(3);
    hold on;
    plot(zlistm, y, '-', 'LineWidth', 1);

    % Calculate model Ksn
    ksnm = ksn(Sm ,DEMm ,Am , m_sp / n_sp);
    ksnm = smooth(Sm, ksnm);
    
    % Plot combined SA
    fig = figure(4);
    hold on
    p = polyfit(log10(SAm.a), log10(SAm.g), 1);
    y = polyval(p, log10(SAm.a));
    hold on;
    plot(SAm.a, 10.^(y), 'DisplayName', ['K = ' erase(list{i}, '.tif')]);
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', 'log')
    
    % Boxplot
    B(1:size(zlistm, 1), i + 1) = zlistm;

    % SS_diff
    SS_diff = DEMr.Z - DEMm.Z;
    fig = figure(1);
    c_max = [-max(SS_diff(:)) max(SS_diff(:))];
    c_man = [-100 100];
    imageschs(DEMr, SS_diff, 'colormap', colormap(flipud(redblue(255))), 'caxis', c_max);
    c = colorbar;                    
    ylabel(c, 'Topographic anomaly relative to steady state (m)');
    title(['Model Grid (K_{sp} = ' erase(list{i}, '.tif') ')']);
    exportgraphics(fig, ['K0_validate_output/SS_diff_' erase(list{i}, '.tif') '.' file_type], 'Resolution', file_res);
    close fig 1;  
    
end

% Section break
disp(' ')

%% Finish figures

% Print status
disp('Exporting final figures...')

% Finish combined pdf
fig = figure(3);
hold on;
xlabel('Elevation (m)');
ylabel('pdf');
legend('Real landscape', 'K_{0l}', 'K_{0h}', 'K_{0}');
grid();
exportgraphics(fig, ['K0_validate_output/pdf_combined.' file_type], 'Resolution', file_res);
close fig 3;

% Finish combined SA
fig = figure(4);
xlabel('Drainage area (m^2)')
ylabel('Slope')
legend();
grid();
exportgraphics(fig, ['K0_validate_output/SA_combined.' file_type], 'Resolution', file_res)
close fig 4;

% Finish Boxplot
fig = figure(5)
boxplot(B, 'Labels', {'Real landscape', 'K_{0l}', 'K_{0h}', 'K_{0}'});
set(gca,'XTickLabel',{'Real landscape', 'K_{0l}', 'K_{0h}', 'K_{0}'});
set(gca, 'TickLabelInterpreter', 'tex')  
ylabel('Elevation (m)');
grid();
exportgraphics(fig, ['K0_validate_output/Boxplot.' file_type], 'Resolution', file_res)
close fig 5;

% Finish Boxchart
fig = figure(6)
boxchart(B, 'MarkerStyle','none');
set(gca,'XTickLabel',{'Real landscape', 'K_{0l}', 'K_{0h}', 'K_{0}'}, 'fontsize', 11);
ylim([300, 800])
ylabel('Elevation (m)')
grid();
exportgraphics(fig, ['K0_validate_output/Boxchart.' file_type], 'Resolution', file_res)
close fig 6;

% Section break
disp(' ')

%% Finalize

 % Print status
disp('Finished!')