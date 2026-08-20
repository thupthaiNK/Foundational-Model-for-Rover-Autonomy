% Purpose: Generate the two real-camera thesis figures that replace the old
%          three-stage sim-to-real cascade bar chart. The cascade chart was
%          built on the superseded 20-image Exp 5b pilot (single 20.0%
%          overall accuracy); both figures here are built from the 200-image
%          terrain_photo_gap_analysis run instead, and report per class
%          because the real-camera failure is class-specific, not uniform.
%            Figure 4.9  real_camera_by_class  -- accuracy per class/source
%            Figure 5.2  sim_to_real_gap       -- AI4Mars vs real camera
%          Uses Visible='off' figures with explicit axes handles throughout,
%          matching make_thesis_figures_3.m (default-visible figures hang
%          under matlab.exe -batch on this WSL setup).
% Inputs:  experiments/results/terrain_photo_gap_analysis_summary.csv
%          experiments/results/dinov2_reg_small_terrain_few_shot.csv
% Outputs: experiments/results/figures/thesis/real_camera_by_class.png(+.pdf)
%          experiments/results/figures/thesis/sim_to_real_gap.png(+.pdf)
% How to run (from WSL):
%   "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch "run('experiments/make_thesis_figures_4.m')"
% Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
resDir = fullfile(scriptDir, 'results');
outDir = fullfile(resDir, 'figures', 'thesis');
if ~exist(outDir, 'dir'); mkdir(outDir); end

realCol = [0.20 0.45 0.70];   % measured on the rover's camera
ctrlCol = [0.65 0.68 0.72];   % AI4Mars control, not a real-camera result

%% Figure 4.9 -- deployed-model accuracy on the rover's own camera, by class
% The AI4Mars rows this figure used to carry were a hand-picked, hand-labelled
% control. The thesis labels AI4Mars images from their pixel masks, not by eye,
% so that control was measured against a different standard from every other
% AI4Mars number here, and it duplicated what the 287-image gold-standard
% result already establishes. It is replaced by that result as a reference line.
fprintf('Figure 4.9 (real camera by class)...\n');

labels = {'Soil', 'Sand', 'Big rock'};
counts = {'13 of 13', '17 of 23', '0 of 36'};
acc    = [100.0, 73.9, 0.0];
ciLo   = [77.2, 53.5, 0.0];
ciHi   = [100.0, 87.5, 9.6];
AI4MARS_REFERENCE = 90.24;

fig = figure('Position', [100 100 700 550], 'Color', 'w', 'Visible', 'off');
ax = axes(fig);
hold(ax, 'on');
yline(ax, AI4MARS_REFERENCE, '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.2);
text(ax, 3.5, AI4MARS_REFERENCE - 3.5, ...
    sprintf('%.2f%% on the AI4Mars test set', AI4MARS_REFERENCE), ...
    'HorizontalAlignment', 'right', 'FontName', 'Arial', 'FontSize', 9, ...
    'Color', [0.45 0.45 0.45]);
for i = 1:numel(acc)
    bar(ax, i, acc(i), 0.55, 'FaceColor', realCol, 'EdgeColor', 'none');
    errorbar(ax, i, acc(i), acc(i) - ciLo(i), ciHi(i) - acc(i), ...
        'Color', [0.1 0.1 0.1], 'LineWidth', 1.1, 'CapSize', 10, 'LineStyle', 'none');
    text(ax, i, ciHi(i) + 5.5, sprintf('%.1f%%', acc(i)), ...
        'HorizontalAlignment', 'center', 'FontName', 'Arial', 'FontSize', 10);
    text(ax, i, -6, counts{i}, 'HorizontalAlignment', 'center', ...
        'FontName', 'Arial', 'FontSize', 9, 'Color', [0.35 0.35 0.35]);
end
set(ax, 'XTick', 1:numel(acc), 'XTickLabel', labels, ...
    'FontName', 'Arial', 'FontSize', 10);
ylabel(ax, 'Classification accuracy / %', 'FontName', 'Arial', 'FontSize', 11);
xlim(ax, [0.4 3.6]);
ylim(ax, [-10 112]);
box(ax, 'on'); grid(ax, 'on');
ax.YGrid = 'on'; ax.XGrid = 'off';
hold(ax, 'off');
save_fig(fig, outDir, 'real_camera_by_class');

%% Figure 5.2 -- AI4Mars against the real camera, per class
fprintf('Figure 5.2 (AI4Mars vs real camera)...\n');

classNames = {'Soil', 'Bedrock', 'Sand', 'Big rock'};
% AI4Mars column: deployed DINOv2+reg ViT-S/14 at 1000 shots
% (dinov2_reg_small_terrain_few_shot.csv, final row). Large rock is not a
% reportable class on the AI4Mars gold-standard test set.
ai4mars = [95.93, 84.78, 87.50, NaN];
% Real camera column: the rover-camera rows of the 200-photograph set. No
% real-world bedrock was photographed, so that cell is genuinely untested.
realcam = [100.0, NaN, 73.9, 0.0];

fig = figure('Position', [100 100 800 550], 'Color', 'w', 'Visible', 'off');
ax = axes(fig);
h = bar(ax, [ai4mars; realcam]', 'grouped', 'EdgeColor', 'none');
h(1).FaceColor = ctrlCol;
h(2).FaceColor = realCol;
hold(ax, 'on');
vals = [ai4mars; realcam];
nanNote = repmat({''}, 2, numel(classNames));
nanNote{1, 4} = 'not a reportable class';
nanNote{2, 2} = 'not photographed';
for g = 1:2
    xs = h(g).XEndPoints;
    for i = 1:numel(classNames)
        if isnan(vals(g, i))
            % "not tested" was wrong for the AI4Mars big-rock cell: that
            % class is not reportable on this test set AND was excluded from
            % the probe's label set, which is a different fact from bedrock
            % simply never having been photographed. Say which is which.
            text(ax, xs(i), 3, nanNote{g, i}, 'Rotation', 90, ...
                'HorizontalAlignment', 'left', 'FontName', 'Arial', ...
                'FontSize', 9, 'Color', [0.45 0.45 0.45]);
        else
            text(ax, xs(i), vals(g, i) + 2, sprintf('%.1f', vals(g, i)), ...
                'HorizontalAlignment', 'center', 'FontName', 'Arial', 'FontSize', 9);
        end
    end
end
set(ax, 'XTickLabel', classNames, 'FontName', 'Arial', 'FontSize', 10);
ylabel(ax, 'Classification accuracy / %', 'FontName', 'Arial', 'FontSize', 11);
legend(ax, {'AI4Mars (NAVCAM)', 'Rover camera'}, 'Location', 'northeast', ...
    'FontName', 'Arial', 'FontSize', 10);
ylim(ax, [0 112]);
box(ax, 'on');
ax.YGrid = 'on'; ax.XGrid = 'off';
hold(ax, 'off');
save_fig(fig, outDir, 'sim_to_real_gap');

fprintf('done\n');

function save_fig(fig, outDir, name)
    ax = findall(fig, 'Type', 'axes');
    for i = 1:numel(ax); ax(i).Toolbar.Visible = 'off'; end
    exportgraphics(fig, fullfile(outDir, [name '.png']), 'Resolution', 300);
    exportgraphics(fig, fullfile(outDir, [name '.pdf']));
    close(fig);
    fprintf('  wrote %s\n', name);
end
