% Purpose: Generate the thesis's key result figures from existing CSVs,
%          following the Cranfield good-figure rubric (axis "quantity / unit",
%          no on-plot title, no spurious connecting lines, thousands space
%          separator, vector output).
% Inputs:  experiments/results/*.csv (already-generated experiment outputs)
% Outputs: experiments/results/figures/thesis/*.png (+ .pdf vector copies)
% How to run (from WSL):
%   "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch "run('experiments/make_thesis_figures.m')"
% Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
resDir = fullfile(scriptDir, 'results');
outDir = fullfile(resDir, 'figures', 'thesis');
if ~exist(outDir, 'dir'); mkdir(outDir); end

paradigmColours = containers.Map( ...
    {'DINO','DINO ensemble','DINO satellite','DINO CNN','Ensemble', ...
     'Positional debiasing','Matryoshka backbone','Contrastive', ...
     'Contrastive sigmoid','Autoregressive','MIM + CLIP teacher', ...
     'Segmentation','Multi-teacher','Depth estimation'}, ...
    { [0.00 0.45 0.70], [0.00 0.45 0.70], [0.00 0.45 0.70], [0.00 0.45 0.70], [0.35 0.35 0.85], ...
      [0.00 0.62 0.45], [0.55 0.35 0.10], [0.90 0.35 0.00], ...
      [0.90 0.35 0.00], [0.80 0.60 0.00], [0.60 0.20 0.40], ...
      [0.50 0.50 0.50], [0.70 0.10 0.10], [0.30 0.30 0.30] });

%% Figure 1 -- 22-model ranking bar chart (1000-shot overall accuracy)
T = readtable(fullfile(resDir, 'model_ranking_1000shot.csv'));
T = sortrows(T, 'overall', 'ascend');   % ascend so barh reads top-to-bottom as best-to-worst
n = height(T);
fig = figure('Position', [100 100 900 700], 'Color', 'w');
hold on;
for i = 1:n
    c = [0.6 0.6 0.6];
    if isKey(paradigmColours, T.paradigm{i}); c = paradigmColours(T.paradigm{i}); end
    barh(i, T.overall(i), 'FaceColor', c, 'EdgeColor', 'none');
end
supAcc = 96.67;
xline(supAcc, '--', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.2);
% Sit the label in the empty strip to the RIGHT of the ceiling line. At
% supAcc-0.3 it printed on top of the dashed line itself and was unreadable.
text(supAcc + 1.4, 1.2, 'supervised ceiling', 'Rotation', 90, ...
    'FontSize', 9, 'HorizontalAlignment', 'left', 'FontName', 'Arial');
yticks(1:n); yticklabels(T.model); ytickangle(0);
set(gca, 'FontSize', 9, 'FontName', 'Arial', 'TickLabelInterpreter', 'none');
xlabel('AI4Mars test accuracy / %', 'FontName', 'Arial', 'FontSize', 11);
xlim([0 100]);
box on; grid on; grid minor;
hold off;
ax = gca; ax.Toolbar.Visible = 'off';
exportgraphics(fig, fullfile(outDir, 'model_ranking_1000shot.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outDir, 'model_ranking_1000shot.pdf'));
close(fig);

%% Figure 2 -- Few-shot accuracy curve: CLIP vs DINOv2 ViT-S vs DINOv2 ViT-L
Tc = readtable(fullfile(resDir, 'few_shot_linear_probe_ViT-B-32.csv'));
Ts = readtable(fullfile(resDir, 'dinov2_reg_small_terrain_few_shot.csv'));
Tl = readtable(fullfile(resDir, 'dinov2_vitl_terrain_few_shot.csv'));

fig = figure('Position', [100 100 800 550], 'Color', 'w');
hold on;
plot(Tc.shots, Tc.overall, '-o', 'Color', [0.90 0.35 0.00], 'LineWidth', 1.5, 'MarkerFaceColor', [0.90 0.35 0.00]);
plot(Ts.shots, Ts.overall, '-s', 'Color', [0.00 0.45 0.70], 'LineWidth', 1.5, 'MarkerFaceColor', [0.00 0.45 0.70]);
plot(Tl.shots, Tl.overall, '-^', 'Color', [0.00 0.20 0.55], 'LineWidth', 1.5, 'MarkerFaceColor', [0.00 0.20 0.55]);
yline(96.67, '--', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.2);
% Headroom above the ceiling line, so the label clears it instead of sitting
% on top of it. At ylim 100 and y=97.3 the text overprinted the dashed line.
text(10, 100.8, 'supervised ceiling (96.67%)', 'FontSize', 9, ...
    'FontName', 'Arial', 'HorizontalAlignment', 'left');
xlabel('labelled training images per class / shots', 'FontName', 'Arial', 'FontSize', 11);
ylabel('AI4Mars overall test accuracy / %', 'FontName', 'Arial', 'FontSize', 11);
legend({'CLIP ViT-B/32', 'DINOv2+reg ViT-S/14 (deployed)', 'DINOv2 ViT-L/14'}, ...
    'Location', 'southeast', 'FontName', 'Arial', 'FontSize', 9, 'Box', 'off');
set(gca, 'FontSize', 10, 'FontName', 'Arial', 'XScale', 'log');
xticks([10 50 100 500 1000]); xticklabels({'10','50','100','500','1 000'});
xlim([9 1100]); ylim([0 106]);
box on; grid on;
hold off;
ax = gca; ax.Toolbar.Visible = 'off';
exportgraphics(fig, fullfile(outDir, 'few_shot_accuracy_curve.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outDir, 'few_shot_accuracy_curve.pdf'));
close(fig);

%% Figure 3 -- Label efficiency curve (balanced + practical regime), with std shading
Tle = readtable(fullfile(resDir, 'label_efficiency_curve.csv'));
bal = Tle(strcmp(Tle.sweep, 'balanced'), :);
prac = Tle(strcmp(Tle.sweep, 'practical'), :);
bal = bal(bal.mean_acc > 0, :);    % drop unrun placeholder rows (mean_acc==0)
prac = prac(prac.mean_acc > 0, :);
% Below the 108-image big-rock ceiling the "practical" regime holds big rock at
% 108 while the other three classes get N, so the rare class is the majority of
% the training set and accuracy collapses (42.0% at N=50). That is a real result
% and is reported in the text, but it is not a regime anyone would choose, and
% plotting it puts the orange curve below the blue one at the same N, which
% reads as a plotting error. The curve therefore starts where the two regimes
% meet, at the ceiling itself.
prac = prac(prac.n_per_class >= 108, :);
bal = sortrows(bal, 'n_per_class');
prac = sortrows(prac, 'n_per_class');

fig = figure('Position', [100 100 800 550], 'Color', 'w');
hold on;
fill([bal.n_per_class; flipud(bal.n_per_class)], ...
     [bal.mean_acc - bal.std_acc; flipud(bal.mean_acc + bal.std_acc)], ...
     [0.00 0.45 0.70], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(bal.n_per_class, bal.mean_acc, '-', 'Color', [0.00 0.45 0.70], 'LineWidth', 1.8);
fill([prac.n_per_class; flipud(prac.n_per_class)], ...
     [prac.mean_acc - prac.std_acc; flipud(prac.mean_acc + prac.std_acc)], ...
     [0.80 0.30 0.00], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(prac.n_per_class, prac.mean_acc, '-', 'Color', [0.80 0.30 0.00], 'LineWidth', 1.8);
xline(108, ':', 'Color', [0.3 0.3 0.3], 'LineWidth', 1);
text(122, 41, 'big rock ceiling (108 images)', 'FontSize', 9, 'FontName', 'Arial');
xlabel('labelled images per class (soil, bedrock, sand)', 'FontName', 'Arial', 'FontSize', 11);
ylabel('AI4Mars overall test accuracy / %', 'FontName', 'Arial', 'FontSize', 11);
legend({'balanced sweep (\pm1 s.d.)', 'balanced sweep mean', ...
        'practical sweep (\pm1 s.d.)', 'practical sweep mean'}, ...
    'Location', 'southeast', 'FontName', 'Arial', 'FontSize', 9, 'Box', 'off');
set(gca, 'FontSize', 10, 'FontName', 'Arial');
% 0-100 squashed the whole story into the top fifth of the axes.
xlim([0 1050]); ylim([35 95]);
box on; grid on;
hold off;
ax = gca; ax.Toolbar.Visible = 'off';
exportgraphics(fig, fullfile(outDir, 'label_efficiency_curve.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outDir, 'label_efficiency_curve.pdf'));
close(fig);

%% Figure 4 -- Coverage-risk Pareto curve (confidence threshold sweep)
Tcr = readtable(fullfile(resDir, 'coverage_risk_curve.csv'));
Tcr = sortrows(Tcr, 'threshold');

fig = figure('Position', [100 100 800 550], 'Color', 'w');
hold on;
plot(Tcr.coverage * 100, Tcr.risk * 100, '-', 'Color', [0.00 0.45 0.70], 'LineWidth', 1.8);
[~, idxKnee] = min(abs(Tcr.threshold - 0.40));
scatter(Tcr.coverage(idxKnee) * 100, Tcr.risk(idxKnee) * 100, 70, [0.85 0.10 0.10], ...
    'filled', 'MarkerEdgeColor', 'k');
text(Tcr.coverage(idxKnee) * 100 - 1.5, Tcr.risk(idxKnee) * 100 - 1.1, ...
    'deployed threshold 0.40', 'FontSize', 9, 'FontName', 'Arial', ...
    'HorizontalAlignment', 'right', 'Color', [0.85 0.10 0.10]);
% Mark a second operating point so the reader can see what abstaining buys.
[~, idxLow] = min(abs(Tcr.coverage - 0.70));
scatter(Tcr.coverage(idxLow) * 100, Tcr.risk(idxLow) * 100, 70, [0.20 0.45 0.20], ...
    'filled', 'MarkerEdgeColor', 'k');
text(Tcr.coverage(idxLow) * 100 + 1.5, Tcr.risk(idxLow) * 100 - 0.9, ...
    'stop on the least confident 30%', 'FontSize', 9, 'FontName', 'Arial', ...
    'Color', [0.20 0.45 0.20]);
xlabel('frames the rover is willing to act on / %', 'FontName', 'Arial', 'FontSize', 11);
ylabel('errors among those frames / %', 'FontName', 'Arial', 'FontSize', 11);
set(gca, 'FontSize', 10, 'FontName', 'Arial');
% The left half of a 0-100 axis held no data at all, squeezing every point
% the reader needs into the right-hand edge.
xlim([50 101]); ylim([0 max(Tcr.risk)*100*1.15]);
box on; grid on;
hold off;
ax = gca; ax.Toolbar.Visible = 'off';
exportgraphics(fig, fullfile(outDir, 'coverage_risk_curve.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outDir, 'coverage_risk_curve.pdf'));
close(fig);

fprintf('Done. Figures written to %s\n', outDir);
