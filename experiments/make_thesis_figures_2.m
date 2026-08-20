% Purpose: Generate the confusion matrix and t-SNE feature-space figures
%          for DINOv2 ViT-L/14 (the best single-encoder model) and the
%          equivalent t-SNE for the deployed model (DINOv2+reg ViT-S/14),
%          following the Cranfield good-figure rubric (no on-plot title,
%          vector output, clean axis labelling).
% Inputs:  experiments/results/confusion_matrix_dinov2_vitl.csv
%          experiments/results/tsne_dinov2_vitl.csv
%          experiments/results/tsne_dinov2_reg_small.csv
%          (produced by experiments/prep_confusion_tsne_data.py and
%          experiments/prep_confusion_tsne_data_deployed.py)
% Outputs: experiments/results/figures/thesis/confusion_matrix_dinov2_vitl.png(+.pdf)
%          experiments/results/figures/thesis/tsne_dinov2_vitl.png(+.pdf)
%          experiments/results/figures/thesis/tsne_dinov2_reg_small.png(+.pdf)
% How to run (from WSL):
%   "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch "run('experiments/make_thesis_figures_2.m')"
% Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
resDir = fullfile(scriptDir, 'results');
outDir = fullfile(resDir, 'figures', 'thesis');
if ~exist(outDir, 'dir'); mkdir(outDir); end

classNames = {'Soil', 'Bedrock', 'Sand'};

%% Figure 5 -- Confusion matrix (DINOv2 ViT-L/14, 1000-shot)
fprintf('Starting figure 5 (confusion matrix)...\n');
Tcm = readtable(fullfile(resDir, 'confusion_matrix_dinov2_vitl.csv'));
cm = [Tcm.Soil, Tcm.Bedrock, Tcm.Sand];
n = size(cm, 1);
rowTotals = sum(cm, 2);
cmPct = 100 * cm ./ rowTotals;

fprintf('  data loaded, opening figure...\n');
% A white-to-black ramp with no colour bar left the reader with no way to know
% what the shading meant. A single-hue blue ramp reads as "more" without
% needing to be learned, and the colour bar states the quantity outright.
blueMap = [linspace(1, 0.05, 256)', linspace(1, 0.25, 256)', linspace(1, 0.45, 256)'];
fig = figure('Position', [100 100 650 600], 'Color', 'w', 'Visible', 'off');
ax0 = axes(fig);
imagesc(ax0, cmPct, [0 100]);
colormap(ax0, blueMap);
fprintf('  imagesc drawn, adding text labels...\n');
hold(ax0, 'on');
for i = 1:n
    for j = 1:n
        if cmPct(i,j) > 55
            txtColour = [1 1 1];
        else
            txtColour = [0 0 0];
        end
        text(ax0, j, i, sprintf('%d (%.1f%%)', cm(i,j), cmPct(i,j)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontName', 'Arial', 'FontSize', 11, 'Color', txtColour);
    end
end
xticks(ax0, 1:n); yticks(ax0, 1:n);
xticklabels(ax0, classNames); yticklabels(ax0, classNames);
xlabel(ax0, 'predicted class', 'FontName', 'Arial', 'FontSize', 11);
ylabel(ax0, 'true class', 'FontName', 'Arial', 'FontSize', 11);
set(ax0, 'FontSize', 10, 'FontName', 'Arial', 'TickLength', [0 0]);
box(ax0, 'on');
cb = colorbar(ax0);
cb.Label.String = 'share of the true class / %';
cb.Label.FontName = 'Arial';
cb.Label.FontSize = 10;
cb.FontName = 'Arial';
cb.FontSize = 9;
hold(ax0, 'off');
fprintf('  labels added, exporting...\n');
exportgraphics(fig, fullfile(outDir, 'confusion_matrix_dinov2_vitl.png'), 'Resolution', 300);
fprintf('  png exported, writing pdf...\n');
exportgraphics(fig, fullfile(outDir, 'confusion_matrix_dinov2_vitl.pdf'));
close(fig);

fprintf('Figure 5 done.\n');

%% Figure 6 -- t-SNE feature-space scatter (DINOv2 ViT-L/14 test-set CLS features)
fprintf('Starting figure 6 (t-SNE scatter)...\n');
Tt = readtable(fullfile(resDir, 'tsne_dinov2_vitl.csv'));
colours = containers.Map({'Soil','Bedrock','Sand'}, ...
    { [0.90 0.60 0.00], [0.45 0.35 0.25], [0.75 0.65 0.30] });
markers = containers.Map({'Soil','Bedrock','Sand'}, {'o','s','^'});

fprintf('  data loaded, opening figure...\n');
fig = figure('Position', [100 100 800 650], 'Color', 'w', 'Visible', 'off');
ax1 = axes(fig);
hold(ax1, 'on');
h = gobjects(1, numel(classNames));
for k = 1:numel(classNames)
    name = classNames{k};
    mask = strcmp(Tt.class_name, name);
    h(k) = scatter(ax1, Tt.x(mask), Tt.y(mask), 45, colours(name), markers(name), 'filled', ...
        'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.3);
end
% Ring the misclassified images. Without this the reader has to take the
% claim that errors fall on the boundaries between neighbouring classes on
% trust; with it, the claim is visible in the plot.
wrong = Tt.correct == 0;
hWrong = scatter(ax1, Tt.x(wrong), Tt.y(wrong), 150, 'o', ...
    'MarkerEdgeColor', [0.85 0.10 0.10], 'LineWidth', 1.4);
fprintf('  points plotted (%d misclassified ringed), adding labels...\n', sum(wrong));
% The axes carry no ticks and t-SNE units are not interpretable, so naming
% them "arbitrary units" added length without adding information. The caption
% carries the warning that inter-cluster distance is not meaningful.
xlabel(ax1, 't-SNE dimension 1', 'FontName', 'Arial', 'FontSize', 11);
ylabel(ax1, 't-SNE dimension 2', 'FontName', 'Arial', 'FontSize', 11);
legend(ax1, [h, hWrong], [classNames, {'misclassified'}], 'Location', 'best', ...
    'FontName', 'Arial', 'FontSize', 10, 'Box', 'off');
set(ax1, 'FontSize', 10, 'FontName', 'Arial', 'XTick', [], 'YTick', []);
box(ax1, 'on');
hold(ax1, 'off');
fprintf('  labels added, exporting...\n');
exportgraphics(fig, fullfile(outDir, 'tsne_dinov2_vitl.png'), 'Resolution', 300);
fprintf('  png exported, writing pdf...\n');
exportgraphics(fig, fullfile(outDir, 'tsne_dinov2_vitl.pdf'));
close(fig);

fprintf('Figure 6 done.\n');

%% Figure 6b -- t-SNE feature-space scatter (deployed model, DINOv2+reg ViT-S/14)
% Same plotting code as Figure 6, pointed at the model actually deployed on
% the Raspberry Pi rather than the best-accuracy comparison model, so the
% report can show what the shipped classifier's own feature space and errors
% look like. Data comes from prep_confusion_tsne_data_deployed.py.
fprintf('Starting figure 6b (t-SNE scatter, deployed model)...\n');
Tt2 = readtable(fullfile(resDir, 'tsne_dinov2_reg_small.csv'));

fprintf('  data loaded, opening figure...\n');
fig = figure('Position', [100 100 800 650], 'Color', 'w', 'Visible', 'off');
ax2 = axes(fig);
hold(ax2, 'on');
h2 = gobjects(1, numel(classNames));
for k = 1:numel(classNames)
    name = classNames{k};
    mask = strcmp(Tt2.class_name, name);
    h2(k) = scatter(ax2, Tt2.x(mask), Tt2.y(mask), 45, colours(name), markers(name), 'filled', ...
        'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.3);
end
wrong2 = Tt2.correct == 0;
hWrong2 = scatter(ax2, Tt2.x(wrong2), Tt2.y(wrong2), 150, 'o', ...
    'MarkerEdgeColor', [0.85 0.10 0.10], 'LineWidth', 1.4);
fprintf('  points plotted (%d misclassified ringed), adding labels...\n', sum(wrong2));
xlabel(ax2, 't-SNE dimension 1', 'FontName', 'Arial', 'FontSize', 11);
ylabel(ax2, 't-SNE dimension 2', 'FontName', 'Arial', 'FontSize', 11);
legend(ax2, [h2, hWrong2], [classNames, {'misclassified'}], 'Location', 'best', ...
    'FontName', 'Arial', 'FontSize', 10, 'Box', 'off');
set(ax2, 'FontSize', 10, 'FontName', 'Arial', 'XTick', [], 'YTick', []);
box(ax2, 'on');
hold(ax2, 'off');
fprintf('  labels added, exporting...\n');
exportgraphics(fig, fullfile(outDir, 'tsne_dinov2_reg_small.png'), 'Resolution', 300);
fprintf('  png exported, writing pdf...\n');
exportgraphics(fig, fullfile(outDir, 'tsne_dinov2_reg_small.pdf'));
close(fig);

fprintf('Done. Figures written to %s\n', outDir);
