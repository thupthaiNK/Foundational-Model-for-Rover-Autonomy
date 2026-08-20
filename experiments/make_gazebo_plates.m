% Purpose: Compose the Gazebo screenshots into the two plates Chapter 3 uses.
%          Plate 1 is the world itself, which no figure in the thesis had ever
%          shown; plate 2 is what the rover's camera sees at each of the five
%          benchmark zones, which is the input DINOv2 actually classifies.
% Inputs:  docs/figures/gazebo_world_views/*.png (capture_gazebo_world_views.py)
%          docs/figures/gazebo_demo_latest/*_zone_view.png (run_frames.sh)
% Outputs: experiments/results/figures/thesis/gazebo_world_plate.png(+.pdf)
%          experiments/results/figures/thesis/gazebo_zone_views.png(+.pdf)
% How to run (from WSL):
%   "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch "run('experiments/make_gazebo_plates.m')"
% Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
repo = fileparts(scriptDir);
outDir = fullfile(scriptDir, 'results', 'figures', 'thesis');
if ~exist(outDir, 'dir'); mkdir(outDir); end
wv = fullfile(repo, 'docs', 'figures', 'gazebo_world_views');
zv = fullfile(repo, 'docs', 'figures', 'gazebo_demo_latest');

%% Plate 1 -- the world: 2x2, overhead + oblique + two rover-eye views
fprintf('World plate...\n');
% Panel (a) is a pre-labelled plan view supplied by the author (matches the
% presentation deck's own plan-view slide) rather than the auto-labelled
% nadir screenshot this script used to generate: the old auto-labels read
% "rock", which the rest of the thesis calls "big rock" (see word_transfer
% README, "Retired numbers"). world_overhead_labeled.png already carries the
% correct "big rock" quadrant label baked in, so panels (a) skips the
% per-quadrant text overlay below.
files = {'world_overhead.png', 'world_oblique.png', 'rock_zone_close.png', 'zone_boundary.png'};
% Report plate 1 panel (a) uses the pre-labelled plan view separately (see
% below); the poster variant further down still uses files{1} unmodified,
% so it is left alone here.
reportFiles = files; reportFiles{1} = 'world_overhead_labeled.png';
labels = {'a', 'b', 'c', 'd'};
fig = figure('Position', [100 100 1180 900], 'Color', 'w', 'Visible', 'off');
t = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:4
    ax = nexttile(t);
    img = imread(fullfile(wv, reportFiles{k}));
    imshow(img, 'Parent', ax);
    text(ax, 0.018, 0.965, ['(' labels{k} ')'], 'Units', 'normalized', ...
        'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'bold', ...
        'Color', 'w', 'BackgroundColor', [0 0 0 0.55], 'Margin', 2, ...
        'VerticalAlignment', 'top');
end
exportgraphics(fig, fullfile(outDir, 'gazebo_world_plate.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outDir, 'gazebo_world_plate.pdf'));
fprintf('  wrote gazebo_world_plate\n');

%% Plate 1b -- poster variant: the overhead and oblique views only.
% Panels (c) and (d) are rover-eye views, which the poster already covers with
% the pipeline and the state machine; dropping them halves the plate's height
% and buys the space the larger poster type needs. The report keeps all four.
fprintf('World plate, poster variant...\n');
figP = figure('Position', [100 100 1500 620], 'Color', 'w', 'Visible', 'off');
tP = tiledlayout(figP, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:2
    ax = nexttile(tP);
    imshow(imread(fullfile(wv, files{k})), 'Parent', ax);
    text(ax, 0.018, 0.965, ['(' labels{k} ')'], 'Units', 'normalized', ...
        'FontName', 'Arial', 'FontSize', 17, 'FontWeight', 'bold', ...
        'Color', 'w', 'BackgroundColor', [0 0 0 0.6], 'Margin', 3, ...
        'VerticalAlignment', 'top');
    if k == 1
        q = {{0.30, 0.22, 'soil'},    {0.72, 0.22, 'sand'}, ...
             {0.30, 0.74, 'bedrock'}, {0.72, 0.74, 'rock'}};
        for m = 1:4
            text(ax, q{m}{1}, q{m}{2}, q{m}{3}, 'Units', 'normalized', ...
                'HorizontalAlignment', 'center', 'FontName', 'Arial', ...
                'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w', ...
                'BackgroundColor', [0 0 0 0.55], 'Margin', 3);
        end
    end
end
exportgraphics(figP, fullfile(outDir, 'gazebo_world_poster.png'), 'Resolution', 300);
fprintf('  wrote gazebo_world_poster\n');

%% Plate 2 -- what the rover camera sees at each of the five benchmark zones
fprintf('Zone-view plate...\n');
zfiles = {'soil_zone_view.png', 'bedrock_zone_view.png', 'sand_zone_view.png', ...
          'rock_cluster_view.png', 'boulder_zone_view.png'};
% (d) and (e) are the SAME terrain type at two different spawn points, 4 m
% apart on the same axis: clip_cascade_gazebo_experiment.py places
% rock_cluster at (2.0, -4.0) and boulder_zone at (2.0, -8.0), both with
% ground truth big_rock, both built from the same sphere primitives. They
% are labelled as two positions rather than two terrains because that is
% what they are; the old "rock cluster" / "boulder field" labels implied a
% distinction the world file does not contain.
znames = {'soil', 'bedrock', 'sand', 'position 1', 'position 2'};
zlab = {'a', 'b', 'c', 'd', 'e'};
% Measured DINOv2 confidence at each position, from fig_gazebo_zones.csv.
% Carried onto (d) and (e) because they are the whole reason the rock zone is
% scored twice: the two positions are the same terrain but do not return the
% same answer, and a reader who cannot see that reasonably asks why one of
% the two images was not simply deleted.
zconf = {'', '', '', '0.33', '0.37'};
fig2 = figure('Position', [100 100 1180 700], 'Color', 'w', 'Visible', 'off');
t2 = tiledlayout(fig2, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
axs = gobjects(1, 5);
for k = 1:5
    ax = nexttile(t2); axs(k) = ax;
    imshow(imread(fullfile(zv, zfiles{k})), 'Parent', ax);
    text(ax, 0.02, 0.95, ['(' zlab{k} ') ' znames{k}], 'Units', 'normalized', ...
        'FontName', 'Arial', 'FontSize', 13, 'FontWeight', 'bold', ...
        'Color', 'w', 'BackgroundColor', [0 0 0 0.62], 'Margin', 3, ...
        'VerticalAlignment', 'top');
    if ~isempty(zconf{k})
        text(ax, 0.98, 0.05, ['confidence ' zconf{k}], 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
            'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'bold', ...
            'Color', 'w', 'BackgroundColor', [0 0 0 0.62], 'Margin', 3);
    end
end
ax = nexttile(t2); axis(ax, 'off');
text(ax, 0.5, 0.74, 'Five views, four terrain types', 'Units', 'normalized', ...
    'HorizontalAlignment', 'center', 'FontName', 'Arial', 'FontSize', 14, ...
    'FontWeight', 'bold', 'Color', [0.15 0.15 0.15]);
text(ax, 0.5, 0.40, {'(d) and (e) are one rock zone', ...
    'scored at two spawn points 4 m', 'apart. Same terrain, but not the', ...
    'same answer: both fall below the', '0.40 threshold and STOP, and at', ...
    'a larger encoder one of the two', 'does not (Section 4.10).'}, ...
    'Units', 'normalized', 'HorizontalAlignment', 'center', ...
    'FontName', 'Arial', 'FontSize', 12, 'Color', [0.28 0.28 0.28]);

% Bracket (d) and (e) together so the pairing is visible before the caption
% is read. Drawn from the two axes' own extents rather than hard-coded, so it
% survives any change to the tile layout.
drawnow;
p4 = get(axs(4), 'Position'); p5 = get(axs(5), 'Position');
x0 = min(p4(1), p5(1)); x1 = max(p4(1)+p4(3), p5(1)+p5(3));
y0 = min(p4(2), p5(2)); y1 = max(p4(2)+p4(4), p5(2)+p5(4));
pad = 0.008;
annotation(fig2, 'rectangle', [x0-pad y0-pad (x1-x0)+2*pad (y1-y0)+2*pad], ...
    'Color', [0.72 0.13 0.13], 'LineWidth', 2.0);
annotation(fig2, 'textbox', [x0-pad y1+pad*0.4 (x1-x0)+2*pad 0.05], ...
    'String', 'one rock zone, scored at two positions', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'FontName', 'Arial', 'FontSize', 12.5, 'FontWeight', 'bold', ...
    'Color', [0.72 0.13 0.13], 'EdgeColor', 'none');
exportgraphics(fig2, fullfile(outDir, 'gazebo_zone_views.png'), 'Resolution', 300);
exportgraphics(fig2, fullfile(outDir, 'gazebo_zone_views.pdf'));
fprintf('  wrote gazebo_zone_views\n');
fprintf('done\n');
