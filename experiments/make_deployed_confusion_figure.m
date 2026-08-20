% Purpose: Generate the confusion matrix figure for the DEPLOYED model
%          (DINOv2+reg ViT-S/14, 1000-shot), the model that actually runs
%          on the Raspberry Pi, following the same Cranfield good-figure
%          rubric as make_thesis_figures_2.m's ViT-L version (no on-plot
%          title, vector output, clean axis labelling). Kept as a separate
%          script rather than folded into make_thesis_figures_2.m, whose
%          own docstring is specifically about the ViT-L comparison model.
% Inputs:  experiments/results/confusion_matrix_dinov2_reg_small.csv
%          (produced by experiments/prep_deployed_confusion_data.py)
% Outputs: experiments/results/figures/thesis/confusion_matrix_dinov2_reg_small.png(+.pdf)
% How to run (from WSL):
%   "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch "run('experiments/make_deployed_confusion_figure.m')"
% Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
resDir = fullfile(scriptDir, 'results');
outDir = fullfile(resDir, 'figures', 'thesis');
if ~exist(outDir, 'dir'); mkdir(outDir); end

classNames = {'Soil', 'Bedrock', 'Sand'};

fprintf('Starting deployed-model confusion matrix figure...\n');
Tcm = readtable(fullfile(resDir, 'confusion_matrix_dinov2_reg_small.csv'));
cm = [Tcm.Soil, Tcm.Bedrock, Tcm.Sand];
n = size(cm, 1);
rowTotals = sum(cm, 2);
cmPct = 100 * cm ./ rowTotals;

fprintf('  data loaded, opening figure...\n');
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
exportgraphics(fig, fullfile(outDir, 'confusion_matrix_dinov2_reg_small.png'), 'Resolution', 300);
fprintf('  png exported, writing pdf...\n');
exportgraphics(fig, fullfile(outDir, 'confusion_matrix_dinov2_reg_small.pdf'));
close(fig);
fprintf('Done.\n');
