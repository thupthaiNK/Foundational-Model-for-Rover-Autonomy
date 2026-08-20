%% make_deck_figures.m
% Purpose: Slide variants of the thesis figures. Two things differ from the
%          report versions, and both are forced by the medium rather than by
%          taste. The report figures are close to square, so on a 16:9 slide
%          they fit to height and leave wide white margins with type too small
%          to read. And they carry the report's per-paradigm palette, whereas
%          a slide states one thing and colours only that.
%          The report figures are not modified.
% Inputs:  experiments/results/model_ranking_1000shot.csv
% Outputs: experiments/results/figures/thesis/*_deck.png (+ .pdf)
% How to run (from WSL):
%   "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch "run('experiments/make_deck_figures.m')"
% Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
resDir = fullfile(scriptDir, 'results');
outDir = fullfile(resDir, 'figures', 'thesis');
if ~exist(outDir, 'dir'); mkdir(outDir); end

FONT  = 'Arial';
NAVY  = [0.047 0.251 0.427];   % #0C406D
BLUE  = [0.102 0.451 0.910];   % #1A73E8
GREEN = [0.118 0.557 0.243];   % #1E8E3E
AMBER = [0.890 0.455 0.000];   % #E37400
RED   = [0.851 0.188 0.145];   % #D93025
GREY  = [0.431 0.431 0.431];   % #6E6E6E
PURPLE= [0.667 0.000 0.667];   % #AA00AA, validated against GREEN at dE 11.4
INK   = [0.102 0.102 0.102];

% 16:9 proportions. The slide box is about 12.1 x 4.9 in, so the figure is
% drawn at that ratio and fills the width instead of being height-limited.
POS = [100 100 1560 620];

    % A local function in a script does not see the script workspace, so the
    % output directory has to be passed in explicitly.
    function saveBoth(fig, name, dir_)
        exportgraphics(fig, fullfile(dir_, [name '.png']), 'Resolution', 200);
        exportgraphics(fig, fullfile(dir_, [name '.pdf']), 'ContentType', 'vector');
        close(fig);
        fprintf('  wrote %s\n', name);
    end

%% ── A: full 22-model ranking, DINO highlighted ──────────────────────────
fprintf('Deck: model ranking (DINO highlighted)...\n');
T = readtable(fullfile(resDir, 'model_ranking_1000shot.csv'));
T = sortrows(T, 'overall', 'ascend');       % barh draws bottom-up
isDino = contains(T.paradigm, 'DINO');

fig = figure('Position', POS, 'Color', 'w', 'Visible', 'off');
ax = axes(fig); hold(ax, 'on');
n = height(T);
isDeployedBar = contains(T.model, 'deployed');
isBest = false(n, 1); isBest(n) = true;   % sorted ascending, so the last is best
for i = 1:n
    if isBest(i);              c = GREEN;
    elseif isDeployedBar(i);   c = PURPLE;
    elseif isDino(i);          c = BLUE;
    else;                      c = GREY;
    end
    barh(ax, i, T.overall(i), 0.72, 'FaceColor', c, 'EdgeColor', 'none');
end
text(ax, T.overall + 0.6, (1:n)', compose('%.1f', T.overall), ...
    'VerticalAlignment', 'middle', 'FontName', FONT, 'FontSize', 11, ...
    'Color', INK);
yticks(ax, 1:n); yticklabels(ax, T.model);
xlim(ax, [0 104]); ylim(ax, [0.3 n + 0.7]);
xlabel(ax, 'AI4Mars test accuracy at 1,000 labels per class / %', ...
    'FontName', FONT, 'FontSize', 13);
set(ax, 'FontName', FONT, 'FontSize', 11, 'Box', 'off', 'TickDir', 'out');
grid(ax, 'on'); ax.GridAlpha = 0.10; ax.YGrid = 'off';
plot(ax, [96.67 96.67], [0.3 n + 0.7], '--', 'Color', GREY, 'LineWidth', 1.4);
text(ax, 96.67, n + 2.15, 'supervised ceiling 96.67%', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
    'FontName', FONT, 'FontSize', 11, 'Color', GREY);

% No boundary line. Every DINO bar is coloured and every other bar is grey,
% so the colour change already marks where rank 11 ends; a line across the
% chart repeated it in ink that carried nothing new.
hBest = barh(ax, NaN, NaN, 'FaceColor', GREEN, 'EdgeColor', 'none');
hDep = barh(ax, NaN, NaN, 'FaceColor', PURPLE, 'EdgeColor', 'none');
hDino = barh(ax, NaN, NaN, 'FaceColor', BLUE, 'EdgeColor', 'none');
hOther = barh(ax, NaN, NaN, 'FaceColor', GREY, 'EdgeColor', 'none');
legend(ax, [hBest hDep hDino hOther], ...
    {'best overall', 'deployed on the rover', 'other DINO variants', ...
     'other pretraining paradigms'}, ...
    'Location', 'southeast', 'FontName', FONT, 'FontSize', 12, 'Box', 'off');
ylim(ax, [0.3 n + 2.6]);
saveBoth(fig, 'model_ranking_deck', outDir);

%% ── B: same ranking, the deployed model highlighted ─────────────────────
fprintf('Deck: model ranking (deployed highlighted)...\n');
isDeployed = contains(T.model, 'deployed');

fig = figure('Position', POS, 'Color', 'w', 'Visible', 'off');
ax = axes(fig); hold(ax, 'on');
for i = 1:n
    if isDeployed(i); c = AMBER; else; c = GREY; end
    barh(ax, i, T.overall(i), 0.72, 'FaceColor', c, 'EdgeColor', 'none');
end
text(ax, T.overall + 0.6, (1:n)', compose('%.1f', T.overall), ...
    'VerticalAlignment', 'middle', 'FontName', FONT, 'FontSize', 11, ...
    'Color', INK);
yticks(ax, 1:n); yticklabels(ax, T.model);
xlim(ax, [0 104]); ylim(ax, [0.3 n + 0.7]);
xlabel(ax, 'AI4Mars test accuracy at 1,000 labels per class / %', ...
    'FontName', FONT, 'FontSize', 13);
set(ax, 'FontName', FONT, 'FontSize', 11, 'Box', 'off', 'TickDir', 'out');
grid(ax, 'on'); ax.GridAlpha = 0.10; ax.YGrid = 'off';
saveBoth(fig, 'model_ranking_deployed_deck', outDir);

%% ── C: best result per paradigm ─────────────────────────────────────────
fprintf('Deck: paradigm summary...\n');
paradigms = unique(T.paradigm, 'stable');
best = zeros(numel(paradigms), 1);
for k = 1:numel(paradigms)
    best(k) = max(T.overall(strcmp(T.paradigm, paradigms{k})));
end
[best, ord] = sort(best, 'ascend');
paradigms = paradigms(ord);
isDinoP = contains(paradigms, 'DINO');

fig = figure('Position', POS, 'Color', 'w', 'Visible', 'off');
ax = axes(fig); hold(ax, 'on');
for i = 1:numel(best)
    if isDinoP(i); c = BLUE; else; c = GREY; end
    barh(ax, i, best(i), 0.66, 'FaceColor', c, 'EdgeColor', 'none');
end
text(ax, best + 0.6, (1:numel(best))', compose('%.2f', best), ...
    'VerticalAlignment', 'middle', 'FontName', FONT, 'FontSize', 13, ...
    'FontWeight', 'bold', 'Color', INK);
yticks(ax, 1:numel(best)); yticklabels(ax, paradigms);
xlim(ax, [0 104]); ylim(ax, [0.4 numel(best) + 0.6]);
xlabel(ax, 'Best AI4Mars accuracy reached by the paradigm / %', ...
    'FontName', FONT, 'FontSize', 13);
set(ax, 'FontName', FONT, 'FontSize', 13, 'Box', 'off', 'TickDir', 'out');
grid(ax, 'on'); ax.GridAlpha = 0.10; ax.YGrid = 'off';
hD = barh(ax, NaN, NaN, 'FaceColor', BLUE, 'EdgeColor', 'none');
hO = barh(ax, NaN, NaN, 'FaceColor', GREY, 'EdgeColor', 'none');
legend(ax, [hD hO], {'DINO self-distillation', 'other paradigms'}, ...
    'Location', 'southeast', 'FontName', FONT, 'FontSize', 12, 'Box', 'off');
saveBoth(fig, 'paradigm_summary_deck', outDir);

%% ── D: curation beats volume, DINOv2 against DINOv3 at matched size ─────
fprintf('Deck: curation against volume...\n');
sizes = {'ViT-S', 'ViT-B', 'ViT-L'};
v2 = [89.90 91.30 93.73];
v3 = [90.20 90.60 92.33];

fig = figure('Position', POS, 'Color', 'w', 'Visible', 'off');
ax = axes(fig); hold(ax, 'on');
h = bar(ax, [v2; v3]', 0.74, 'EdgeColor', 'none');
h(1).FaceColor = BLUE;    % DINOv2, 142M curated
h(2).FaceColor = GREY;    % DINOv3, 1.69B web-crawled
for k = 1:3
    text(ax, k - 0.19, v2(k) + 0.35, sprintf('%.2f', v2(k)), ...
        'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', 13, ...
        'FontWeight', 'bold', 'Color', INK);
    text(ax, k + 0.19, v3(k) + 0.35, sprintf('%.2f', v3(k)), ...
        'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', 13, ...
        'Color', INK);
end
% The one place DINOv3 wins is the point the slide must not hide. Putting it
% in the tick label attaches it to the pair it describes; a free-floating
% annotation at the axis floor did not.
xticks(ax, 1:3);
xticklabels(ax, {'ViT-S  (DINOv3 wins)', 'ViT-B', 'ViT-L'});
ylim(ax, [86 95]);
ylabel(ax, 'AI4Mars accuracy / %', 'FontName', FONT, 'FontSize', 13);
legend(ax, {'DINOv2, 142M curated images', 'DINOv3, 1.69B web-crawled images'}, ...
    'Location', 'northwest', 'FontName', FONT, 'FontSize', 12, 'Box', 'off');
set(ax, 'FontName', FONT, 'FontSize', 13, 'Box', 'off', 'TickDir', 'out');
grid(ax, 'on'); ax.GridAlpha = 0.10; ax.XGrid = 'off';
saveBoth(fig, 'curation_vs_volume_deck', outDir);

fprintf('Deck figures done.\n');
