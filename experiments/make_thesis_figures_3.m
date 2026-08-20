% Purpose: Generate 6 more thesis figures (sim-to-real gap, accuracy vs
%          speed trade-off, backbone scaling, RPi/PC deployment comparison,
%          Gazebo 5-zone results, six-paradigm summary), following the
%          Cranfield good-figure rubric. Uses Visible='off' figures with
%          explicit axes handles throughout (default-visible figures +
%          colorbar/axis square hung indefinitely under matlab.exe -batch
%          on this WSL setup in an earlier script -- see
%          make_thesis_figures_2.m's header comment).
% Inputs:  experiments/results/fig_*.csv, e3_model_latency_benchmark.csv
% Outputs: experiments/results/figures/thesis/*.png(+.pdf)
% How to run (from WSL):
%   "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch "run('experiments/make_thesis_figures_3.m')"
% Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
resDir = fullfile(scriptDir, 'results');
outDir = fullfile(resDir, 'figures', 'thesis');
if ~exist(outDir, 'dir'); mkdir(outDir); end

%% Figure 7 -- superseded
% The three-stage sim-to-real cascade bar chart that used to live here was
% built on the 20-image Exp 5b pilot's single 20.0%% overall accuracy. That
% pilot is superseded by the 200-image per-class analysis, so sim_to_real_gap
% is now produced by make_thesis_figures_4.m instead. Do not re-add this
% block: it would overwrite the current figure with the old numbers.

%% Figure 8 -- Accuracy vs inference speed trade-off
fprintf('Figure 8 (accuracy vs speed)...\n');
T = readtable(fullfile(resDir, 'fig_accuracy_vs_speed.csv'));
fig = figure('Position', [100 100 800 600], 'Color', 'w', 'Visible', 'off');
ax = axes(fig);
hold(ax, 'on');
notDep = T.deployed == 0;
dep = T.deployed == 1;
% Handles are kept so the legend names the two colours. Without it the reader
% has to infer from a caption what red means, and on a slide there is no
% caption to read.
hOther = scatter(ax, T.ms_per_img(notDep), T.accuracy(notDep), 70, [0.20 0.45 0.70], 'filled', 'MarkerEdgeColor', [0.1 0.1 0.1]);
hDep   = scatter(ax, T.ms_per_img(dep), T.accuracy(dep), 110, [0.85 0.10 0.10], 'filled', '^', 'MarkerEdgeColor', 'k');
% Explicit per-point offsets (not row-parity alternation -- parity
% happened to push the two closest pairs of points toward each other
% instead of apart) so the two near-duplicate accuracy clusters
% (~90.2% ViT-S pair, ~91/90.6% ViT-B pair) read as separate labels.
% Order matches fig_accuracy_vs_speed.csv: CLIP, DINOv2 S/B/L, DINOv3 S/B/L.
vaList = {'top', 'bottom', 'bottom', 'bottom', 'top', 'top', 'top'};
dyList = [-0.3, 0.3, 0.3, 0.2, -0.3, -0.3, -0.2];
for i = 1:height(T)
    % Colour each label to match its marker. The deployed model sits almost on
    % top of DINOv3 ViT-S/16, so two black labels near one pair of points left
    % the reader unable to tell which was which.
    if T.deployed(i) == 1
        lblCol = [0.85 0.10 0.10];
    else
        lblCol = [0.15 0.30 0.50];
    end
    text(ax, T.ms_per_img(i) * 1.05, T.accuracy(i) + dyList(i), T.model{i}, ...
        'FontName', 'Arial', 'FontSize', 9, 'VerticalAlignment', vaList{i}, ...
        'Color', lblCol);
end
set(ax, 'XScale', 'log', 'FontSize', 10, 'FontName', 'Arial');
xticks(ax, [200 300 500 1000 2000]);
xticklabels(ax, {'200', '300', '500', '1000', '2000'});
xlabel(ax, 'feature-extraction latency / ms per image', 'FontName', 'Arial', 'FontSize', 11);
ylabel(ax, 'AI4Mars overall accuracy / %', 'FontName', 'Arial', 'FontSize', 11);
xlim(ax, [150 2500]);
ylim(ax, [85 96]);
box(ax, 'on'); grid(ax, 'on');
lg = legend(ax, [hDep hOther], ...
    {'deployed on the rover: DINOv2+reg ViT-S/14', 'other encoders tested'}, ...
    'Location', 'southeast', 'FontName', 'Arial', 'FontSize', 10);
lg.Box = 'on';
hold(ax, 'off');
save_fig(fig, outDir, 'accuracy_vs_speed');

%% Figure 9 -- Backbone scaling (DINOv2 vs DINOv3, ViT-S/B/L)
fprintf('Figure 9 (backbone scaling)...\n');
T = readtable(fullfile(resDir, 'fig_backbone_scaling.csv'));
fig = figure('Position', [100 100 800 600], 'Color', 'w', 'Visible', 'off');
ax = axes(fig);
hold(ax, 'on');
v2 = strcmp(T.family, 'DINOv2');
v3 = strcmp(T.family, 'DINOv3');
plot(ax, T.params_m(v2), T.accuracy(v2), '-o', 'Color', [0.00 0.45 0.70], ...
    'LineWidth', 1.8, 'MarkerFaceColor', [0.00 0.45 0.70], 'MarkerSize', 8);
plot(ax, T.params_m(v3), T.accuracy(v3), '-s', 'Color', [0.80 0.30 0.00], ...
    'LineWidth', 1.8, 'MarkerFaceColor', [0.80 0.30 0.00], 'MarkerSize', 8);
set(ax, 'XScale', 'log', 'FontSize', 10, 'FontName', 'Arial');
% A log axis labelled only "10^2" gave no way to read off the model sizes.
xticks(ax, [22 86 304]);
xticklabels(ax, {'22 (ViT-S)', '86 (ViT-B)', '304 (ViT-L)'});
xlabel(ax, 'encoder size / million parameters', 'FontName', 'Arial', 'FontSize', 11);
ylabel(ax, 'AI4Mars overall accuracy / %', 'FontName', 'Arial', 'FontSize', 11);
legend(ax, {'DINOv2', 'DINOv3'}, 'Location', 'southeast', 'FontName', 'Arial', 'FontSize', 10, 'Box', 'off');
xlim(ax, [18 350]);
ylim(ax, [88 95]);
box(ax, 'on'); grid(ax, 'on');
hold(ax, 'off');
save_fig(fig, outDir, 'backbone_scaling');

%% Figure 10 -- Raspberry Pi / PC deployment comparison (RAM and latency)
% Two separate simple figures rather than one tiledlayout figure: an
% earlier two-axes-per-figure attempt hung indefinitely under
% matlab.exe -batch on this WSL setup (same failure class as the
% colorbar/axis-square hang documented in make_thesis_figures_2.m).
fprintf('Figure 10a (deployment latency)...\n');
rawTab = readcell(fullfile(resDir, 'e3_model_latency_benchmark.csv'));
hdr = rawTab(1, :);
modelCol = find(strcmp(hdr, 'model'));
msCol = find(strcmp(hdr, 'mean_ms'));
ramCol = find(strcmp(hdr, 'ram_mb'));
data = rawTab(2:end, :);
modelNames = string(data(:, modelCol));
msVals = cell2mat(data(:, msCol));
ramVals = zeros(size(data, 1), 1);
for i = 1:size(data, 1)
    v = data{i, ramCol};
    if ischar(v) || isstring(v)
        v = str2double(strrep(v, '~', ''));
    end
    ramVals(i) = v;
end

fig = figure('Position', [100 100 700 550], 'Color', 'w', 'Visible', 'off');
ax = axes(fig);
bar(ax, categorical(modelNames, modelNames), msVals, 'FaceColor', [0.20 0.45 0.70]);
set(ax, 'YScale', 'log', 'FontSize', 10, 'FontName', 'Arial');
ylabel(ax, 'PC latency / ms per image (log scale)', 'FontName', 'Arial', 'FontSize', 11);
box(ax, 'on'); grid(ax, 'on');
save_fig(fig, outDir, 'deployment_latency');

fprintf('Figure 10b (deployment RAM)...\n');
fig = figure('Position', [100 100 700 550], 'Color', 'w', 'Visible', 'off');
ax = axes(fig);
bar(ax, categorical(modelNames, modelNames), ramVals, 'FaceColor', [0.80 0.30 0.00]);
set(ax, 'YScale', 'log', 'FontSize', 10, 'FontName', 'Arial');
ylabel(ax, 'peak RAM / MB (log scale)', 'FontName', 'Arial', 'FontSize', 11);
box(ax, 'on'); grid(ax, 'on');
save_fig(fig, outDir, 'deployment_ram');

%% Figure 11 -- Gazebo 5-zone traversability result
% Plotted as two separate bar() series (correct vs. uncertain) at their
% own x-positions rather than one bar() with per-bar 'flat' CData -- the
% CData approach hung indefinitely under matlab.exe -batch on this WSL
% setup, the same failure class noted elsewhere in this script.
fprintf('Figure 11 (Gazebo zones)...\n');
T = readtable(fullfile(resDir, 'fig_gazebo_zones.csv'));
n = height(T);
correctMask = T.terrain_correct == 1;
confCorrect = nan(n, 1); confCorrect(correctMask) = T.confidence(correctMask);
confUncertain = nan(n, 1); confUncertain(~correctMask) = T.confidence(~correctMask);

fig = figure('Position', [100 100 850 550], 'Color', 'w', 'Visible', 'off');
ax = axes(fig);
hold(ax, 'on');
bar(ax, 1:n, confCorrect, 'FaceColor', [0.00 0.55 0.30]);
bar(ax, 1:n, confUncertain, 'FaceColor', [0.55 0.55 0.55]);
plot(ax, [0.5 n + 0.5], [0.40 0.40], '--', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.2);
text(ax, 0.6, 0.44, 'confidence threshold, T = 0.40', 'FontName', 'Arial', 'FontSize', 9);
for i = 1:n
    % Below-threshold bars take their label BELOW the bar top. Above it, the
    % label lands on the dashed threshold line and cuts it (the rightmost bar
    % did exactly that at 0.369 + 0.04 = 0.409).
    if T.confidence(i) < 0.40
        yLab = T.confidence(i) - 0.045; vAlign = 'top';
    else
        yLab = T.confidence(i) + 0.03; vAlign = 'bottom';
    end
    text(ax, i, yLab, sprintf('%s -> %s', T.ground_truth{i}, T.prediction{i}), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', vAlign, ...
        'FontName', 'Arial', 'FontSize', 8);
end
% The x labels must match Figure 3.5's panel labels. The CSV keeps the world
% file's own object names (rock_cluster, boulder_zone), which read as two
% different terrains; they are one zone scored at two positions.
zoneLabels = strrep(strrep(T.zone, 'Rock cluster', 'Rock zone, pos. 1'), ...
                    'Boulder zone', 'Rock zone, pos. 2');
xticks(ax, 1:n); xticklabels(ax, zoneLabels); xtickangle(ax, 0);
ylabel(ax, 'classifier confidence', 'FontName', 'Arial', 'FontSize', 11);
set(ax, 'FontSize', 10, 'FontName', 'Arial');
xlim(ax, [0.5 n + 0.5]);
ylim(ax, [0 1]);
box(ax, 'on'); grid(ax, 'on');
hold(ax, 'off');
save_fig(fig, outDir, 'gazebo_zone_results');

%% Figure 12 -- Six-paradigm summary
fprintf('Figure 12 (paradigm summary)...\n');
T = readtable(fullfile(resDir, 'fig_paradigm_summary.csv'));
T = sortrows(T, 'accuracy', 'descend');
fig = figure('Position', [100 100 900 550], 'Color', 'w', 'Visible', 'off');
ax = axes(fig);
n = height(T);
% Flip labels AND values together so category order and bar heights stay
% paired (categorical()'s 2nd arg only reorders the display, it does not
% reorder the data -- flipping just the value vector while leaving the
% category vector unflipped desynced bar height from bar label).
labelsFlip = flip(T.paradigm);
modelFlip = flip(T.best_model);
accFlip = flip(T.accuracy);
cats = categorical(labelsFlip, labelsFlip);
% best paradigm (row 1, pre-sort, highest accuracy) ends up last in the
% flipped order, i.e. top of the barh -- isolate it into its own series
% instead of using a bar() object's 'flat' CData -- that property hung
% indefinitely under matlab.exe -batch on this WSL setup (same failure
% class as Figure 11).
accRest = accFlip; accRest(n) = NaN;
accBest = nan(n, 1); accBest(n) = accFlip(n);
barh(ax, cats, accRest, 'FaceColor', [0.55 0.55 0.55]);
hold(ax, 'on');
barh(ax, cats, accBest, 'FaceColor', [0.00 0.45 0.70]);
for i = 1:n
    text(ax, accFlip(i) + 1, i, modelFlip{i}, ...
        'FontName', 'Arial', 'FontSize', 9, 'VerticalAlignment', 'middle');
end
xlabel(ax, 'best 1000-shot AI4Mars accuracy in this paradigm / %', 'FontName', 'Arial', 'FontSize', 11);
set(ax, 'FontSize', 10, 'FontName', 'Arial');
xlim(ax, [0 100]);
box(ax, 'on'); grid(ax, 'on');
hold(ax, 'off');
save_fig(fig, outDir, 'paradigm_summary');

fprintf('Done. Figures written to %s\n', outDir);

function save_fig(fig, outDir, name)
    ax = findall(fig, 'Type', 'axes');
    for i = 1:numel(ax); ax(i).Toolbar.Visible = 'off'; end
    exportgraphics(fig, fullfile(outDir, [name '.png']), 'Resolution', 300);
    exportgraphics(fig, fullfile(outDir, [name '.pdf']));
    close(fig);
    fprintf('  wrote %s\n', name);
end
