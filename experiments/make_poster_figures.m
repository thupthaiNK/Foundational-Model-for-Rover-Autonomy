% Purpose: Draw the two figure variants the A1 poster needs and the thesis
%          does not. Both exist because a poster is read standing up at about
%          1.5 m, where the report versions stop working:
%            1. model_ranking_poster  -- the 22-model survey cut to 10 rows.
%               All 22 bars at poster column width give 4 mm of vertical
%               space per label.
%            2. reactive_fsm_poster   -- the state machine without the
%               five-line trigger footnote, which is unreadable at this size
%               and is detail a poster reader does not need.
%          Neither replaces its thesis counterpart; the report keeps the full
%          versions.
% Inputs:  experiments/results/model_ranking_1000shot.csv
% Outputs: experiments/results/figures/thesis/model_ranking_poster.png(+.pdf)
%          experiments/results/figures/thesis/reactive_fsm_poster.png(+.pdf)
% How to run (from WSL):
%   "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch "run('experiments/make_poster_figures.m')"
% Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
resDir = fullfile(scriptDir, 'results');
outDir = fullfile(resDir, 'figures', 'thesis');
if ~exist(outDir, 'dir'); mkdir(outDir); end

FONT = 'Arial';
NAVY  = [0.047 0.251 0.427];
ACCENT= [0.929 0.490 0.192];
GREY  = [0.62 0.62 0.62];
INK   = [0.12 0.12 0.12];

%% ── Poster figure A: 10-model ranking ───────────────────────────────────
fprintf('Poster model ranking...\n');
T = readtable(fullfile(resDir, 'model_ranking_1000shot.csv'));

% Ten rows: the top six, then the best remaining representative of each of
% the paradigms the top six do not already cover. That keeps the poster's
% claim ("DINO leads, the others do not") checkable rather than cherry-picked.
% Six bars, not eleven. At poster size eleven rows gave about 4 mm per label,
% and most of them repeated the same message. These six are the smallest set
% that still lets a reader CHECK the claim rather than take it on trust: the
% best overall, the best single encoder, the model actually deployed, the
% closest non-DINO paradigm, a contrastive baseline, and the worst result.
% Dropping the non-DINO bars entirely would leave a chart of DINO models that
% proves nothing about DINO.
wanted = {'Ensemble B (DINOv2 L+B)', 'DINOv2 ViT-L/14', ...
          'DINOv2+reg ViT-S (deployed)', 'Franca-RASA', ...
          'CLIP ViT-B/32', 'RADIO-B'};
keepIdx = [];
for k = 1:numel(wanted)
    hit = find(strcmp(strtrim(T.model), wanted{k}), 1, 'first');
    if isempty(hit); error('poster ranking: model not found: %s', wanted{k}); end
    keepIdx(end+1) = hit; %#ok<SAGROW>
end
S = T(keepIdx, :);
[~, ord] = sort(S.overall, 'ascend');   % barh draws bottom-up
S = S(ord, :);

isDino = contains(S.paradigm, 'DINO');
isDeployed = contains(S.model, 'deployed') | contains(S.model, 'reg ViT-S');

fig = figure('Position', [100 100 1250 720], 'Color', 'w', 'Visible', 'off');
ax = axes(fig); hold(ax, 'on');
n = height(S);
for i = 1:n
    if isDeployed(i);   c = ACCENT;
    elseif isDino(i);   c = NAVY;
    else;               c = GREY;
    end
    barh(ax, i, S.overall(i), 0.68, 'FaceColor', c, 'EdgeColor', 'none');
    text(ax, S.overall(i) + 0.6, i, sprintf('%.1f%%', S.overall(i)), ...
        'VerticalAlignment', 'middle', 'FontName', FONT, 'FontSize', 19, ...
        'FontWeight', 'bold', 'Color', INK);
end
yticks(ax, 1:n); yticklabels(ax, S.model);
xlim(ax, [0 104]); ylim(ax, [0.4 n + 1.9]);
xlabel(ax, '1000-shot AI4Mars accuracy / %', 'FontName', FONT, 'FontSize', 19);
set(ax, 'FontName', FONT, 'FontSize', 18, 'Box', 'off', 'TickDir', 'out');
grid(ax, 'on'); ax.GridAlpha = 0.12; ax.YGrid = 'off';

% Supervised ceiling, so the bars are read against something.
plot(ax, [96.67 96.67], [0.4 n + 0.75], '--', 'Color', [0.35 0.35 0.35], ...
    'LineWidth', 1.6);
% Horizontal, in the clear band between the top bar and the key. Rotated
% alongside the line it crossed the value labels of every bar it passed.
text(ax, 104, n + 0.62, 'supervised ceiling 96.67%', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
    'FontName', FONT, 'FontSize', 16, 'Color', [0.35 0.35 0.35]);

% Key as one horizontal row ABOVE the bars. Placed inside the axes at any x
% it lands on top of a bar, because every bar starts at zero and the longest
% reaches 94%: there is no empty region inside this plot to put it in.
yk = n + 1.35; xk = 1;
cols = {NAVY, ACCENT, GREY};
labs = {'DINO self-distillation', 'deployed on the rover', 'other paradigms'};
for k = 1:3
    patch(ax, 'XData', xk + [0 2.6 2.6 0], 'YData', yk + [-0.22 -0.22 0.22 0.22], ...
        'FaceColor', cols{k}, 'EdgeColor', 'none');
    text(ax, xk + 3.6, yk, labs{k}, 'VerticalAlignment', 'middle', ...
        'FontName', FONT, 'FontSize', 16, 'Color', INK);
    xk = xk + 34;
end
hold(ax, 'off');
exportgraphics(fig, fullfile(outDir, 'model_ranking_poster.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outDir, 'model_ranking_poster.pdf'));
fprintf('  wrote model_ranking_poster (%d models)\n', n);

%% ── Poster figure B: accuracy against latency, at poster type sizes ─────
% Same data and same layout as Figure 4.8, redrawn with labels and axes sized
% for a poster read at about 1.5 m. The report version's 9 pt point labels
% disappear at this placement width.
fprintf('Poster accuracy-vs-latency...\n');
A = readtable(fullfile(resDir, 'fig_accuracy_vs_speed.csv'));
fig = figure('Position', [100 100 1250 900], 'Color', 'w', 'Visible', 'off');
ax = axes(fig); hold(ax, 'on');
notDep = A.deployed == 0; dep = A.deployed == 1;
scatter(ax, A.ms_per_img(notDep), A.accuracy(notDep), 220, [0.20 0.45 0.70], ...
    'filled', 'MarkerEdgeColor', [0.1 0.1 0.1], 'LineWidth', 1.2);
scatter(ax, A.ms_per_img(dep), A.accuracy(dep), 460, ACCENT, 'filled', '^', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
vaList = {'top', 'bottom', 'bottom', 'bottom', 'top', 'top', 'top'};
dyList = [-0.42, 0.42, 0.42, 0.30, -0.42, -0.42, -0.30];
for i = 1:height(A)
    if A.deployed(i) == 1; lblCol = ACCENT; else; lblCol = [0.15 0.30 0.50]; end
    text(ax, A.ms_per_img(i) * 1.06, A.accuracy(i) + dyList(i), A.model{i}, ...
        'FontName', FONT, 'FontSize', 17, 'FontWeight', 'bold', ...
        'VerticalAlignment', vaList{i}, 'Color', lblCol);
end
set(ax, 'XScale', 'log', 'FontSize', 18, 'FontName', FONT);
xticks(ax, [200 300 500 1000 2000]);
xticklabels(ax, {'200', '300', '500', '1000', '2000'});
xlabel(ax, 'feature-extraction latency / ms per image', 'FontName', FONT, 'FontSize', 20);
ylabel(ax, 'AI4Mars overall accuracy / %', 'FontName', FONT, 'FontSize', 20);
xlim(ax, [150 3000]); ylim(ax, [85 96.5]);
grid(ax, 'on'); ax.GridAlpha = 0.12; box(ax, 'on');
text(ax, 165, 95.6, 'deployed on the rover', 'FontName', FONT, 'FontSize', 17, ...
    'FontWeight', 'bold', 'Color', ACCENT);
hold(ax, 'off');
exportgraphics(fig, fullfile(outDir, 'accuracy_vs_speed_poster.png'), 'Resolution', 300);
fprintf('  wrote accuracy_vs_speed_poster\n');

fprintf('done\n');
