% Purpose: Draw the two structural diagrams Chapter 3 needs: the
%          perception-to-motion pipeline (Figure 3.3), and the
%          reactive-explorer state machine (Figure 3.6). Both are drawn from
%          the shipped code, not from the design intent -- the FSM states and
%          transitions are those in
%          ros2_ws/src/fm_perception/fm_perception/reactive_explorer_node.py
%          as of the 2026-07-29 revision that removed all reverse motion.
%
%          Layout rules, after two rounds of review:
%          1. Every connector is orthogonal. No diagonal crosses the drawing.
%          2. No connector crosses another connector or passes through a box.
%          3. No label sits on a line. Labels for horizontal connectors go
%             above them, labels for vertical connectors go beside them, and
%             nothing uses a white knockout box to hide the line underneath.
%          4. Mission-ending transitions collect onto one bus rather than
%             being routed individually across the figure. This is also what
%             keeps the state machine wider than it is tall, which is what
%             lets it share a page with body text.
%          5. Each box family has a saturated border in its own hue over a
%             pale fill of the same hue, so the grouping survives greyscale
%             printing and stays legible at the 158 mm placement width.
%          6. An arrowhead means "the transition ends here". Legs feeding a
%             shared bus are drawn with lineonly(), no head, so the only
%             heads on the red bus are the four tags' destination, FAILSAFE.
%
%          FSM transitions, re-derived from the node source 2026-08-09 after
%          a review question, and corrected: the previous version of this
%          figure had STARTUP_SWEEP returning to STARTUP_CHECK and
%          TURN_TO_HEADING returning to MONITORING. Both are the wrong way
%          round in the code, and two transitions were missing entirely.
%          What the source actually does:
%            init                -> STARTUP_CHECK                    (l. 434)
%            STARTUP_CHECK       -> STARTUP_SWEEP    first entry     (l. 591)
%            STARTUP_CHECK       -> MONITORING       facing it       (l. 717)
%            STARTUP_CHECK       -> TURN_TO_HEADING  turn needed     (l. 721)
%            STARTUP_SWEEP       -> MONITORING       facing winner   (l. 922)
%            STARTUP_SWEEP       -> TURN_TO_HEADING  via _commit_to_yaw
%            MONITORING          -> TURN_TO_HEADING  hazard held     (l.1006)
%            TURN_TO_HEADING     -> STARTUP_CHECK    turn arrived    (l.1022)
%            any state           -> FAILSAFE         see foot of fig
% Inputs:  none (the layout is hand-specified here)
% Outputs: experiments/results/figures/thesis/fm_pipeline.png(+.pdf)
%          experiments/results/figures/thesis/reactive_fsm.png(+.pdf)
% How to run (from WSL):
%   "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch "run('experiments/make_thesis_diagrams.m')"
% Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
outDir = fullfile(scriptDir, 'results', 'figures', 'thesis');
if ~exist(outDir, 'dir'); mkdir(outDir); end

FONT = 'Arial';

% Palette: pale fill, saturated same-hue border, near-black body text. The
% previous version used grey text on pale fills with grey borders throughout,
% which is the main reason it read as washed out at print size.
sensorFill = [0.85 0.91 0.97];  sensorEdge = [0.17 0.40 0.63];
percFill   = [0.84 0.93 0.85];  percEdge   = [0.13 0.45 0.26];
ctrlFill   = [1.00 0.93 0.78];  ctrlEdge   = [0.72 0.47 0.05];
vetoFill   = [0.98 0.87 0.86];  vetoEdge   = [0.68 0.22 0.19];
actFill    = [0.91 0.91 0.93];  actEdge    = [0.36 0.36 0.42];
failFill   = [0.97 0.85 0.84];  failEdge   = [0.68 0.22 0.19];

INK   = [0.12 0.12 0.12];   % connectors and box titles
LABEL = [0.16 0.16 0.16];   % edge labels
MUTED = [0.26 0.26 0.26];   % box subtitles
RED   = [0.72 0.13 0.13];   % veto and terminal transitions

% Type sizes. The diagrams are placed at 158 mm, so these are deliberately
% large on the canvas; at 13 pt the previous version reproduced too small to
% read comfortably on the page.
TITLE_PT = 16.5; BOX_PT = 14.5; SUB_PT = 12.5; EDGE_PT = 12.5; KEY_PT = 12;

%% ── Figure 3.3: perception-to-motion pipeline ───────────────────────────
fprintf('Pipeline diagram...\n');
fig = figure('Position', [100 100 1340 950], 'Color', 'w', 'Visible', 'off');
ax = axes(fig, 'Position', [0 0 1 1]); hold(ax, 'on');
axis(ax, [0 138 0 96]); axis(ax, 'off');

text(ax, 3, 92.6, 'Perception to motion: the foundation model decides where the rover may drive', ...
    'FontName', FONT, 'FontSize', TITLE_PT, 'FontWeight', 'bold', 'Color', INK);

% ── Band 1: the foundation-model path, left to right along the top.
drawbox(ax,  2, 76, 28, 13, sensorFill, sensorEdge, 'Camera',              '640x480, 10 Hz, 80 deg');
drawbox(ax, 44, 76, 36, 13, percFill,   percEdge,   'DINOv2+reg ViT-S/14', 'frozen encoder, 384-d CLS');
drawbox(ax, 94, 76, 28, 13, percFill,   percEdge,   'Linear probe',        '4 classes, incl. big rock');
arrowH(ax, 30, 44, 82.5, 'image\_raw', INK, LABEL, EDGE_PT);
arrowH(ax, 80, 94, 82.5, 'features',   INK, LABEL, EDGE_PT);
arrowV(ax, 105, 76, 70, 'class + confidence', 'right', INK, LABEL, EDGE_PT);

% ── Band 2: the decision, and the speed it produces.
drawbox(ax, 38, 42, 70, 28, ctrlFill, ctrlEdge, '', '');
text(ax, 73, 66.4, 'Traversability controller', 'HorizontalAlignment', 'center', ...
    'FontName', FONT, 'FontSize', BOX_PT, 'FontWeight', 'bold', 'Color', INK);
text(ax, 73, 62.6, 'a confidence below 0.40 means the class is not trusted', ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', SUB_PT, 'Color', MUTED);
cx  = [44 74 102];
al  = {'left', 'center', 'right'};
hdr = {'terrain class', 'state', 'wheel speed'};
for k = 1:3
    text(ax, cx(k), 58.6, hdr{k}, 'HorizontalAlignment', al{k}, 'FontName', FONT, ...
        'FontSize', SUB_PT, 'FontWeight', 'bold', 'Color', ctrlEdge);
end
plot(ax, [43 103], [56.6 56.6], '-', 'Color', ctrlEdge, 'LineWidth', 1.0);
rows = {'soil',      'SAFE',    '0.10 m/s'; ...
        'sand',      'CAUTION', '0.05 m/s'; ...
        'bedrock',   'HAZARD',  '0.03 m/s'; ...
        'big rock',  'HAZARD',  '0.00 m/s'; ...
        'uncertain', 'STOP',    '0.00 m/s'};
% Five rows in a box that was drawn for four. Starting 0.9 higher and
% stepping 2.5 instead of 3.3 keeps the last row clear of the box edge;
% the previous 2.8 step put 'uncertain' on the boundary itself.
for r = 1:5
    yy = 53.1 - 2.5*(r-1);
    for k = 1:3
        text(ax, cx(k), yy, rows{r,k}, 'HorizontalAlignment', al{k}, ...
            'FontName', FONT, 'FontSize', SUB_PT, 'Color', INK);
    end
end

% ── Band 3: the veto-only channels and the recovery layer they feed.
drawbox(ax,  3, 27, 24, 13, sensorFill, sensorEdge, 'LiDAR', '360 deg scan');
drawbox(ax,  3, 11, 24, 13, sensorFill, sensorEdge, 'IMU',   'tilt, gated');
drawbox(ax, 33, 27, 31, 13, vetoFill,   vetoEdge,   'Proximity guard', 'stop within 0.40 m');
drawbox(ax, 33, 11, 31, 13, vetoFill,   vetoEdge,   'Slope fusion',    'CAUTION 10, STOP 20 deg');
drawbox(ax, 76, 15, 26, 21, ctrlFill,   ctrlEdge,   'Reactive explorer', 'state machine, Figure 3.6');
arrowH(ax, 27, 33, 33.5, '', INK, LABEL, EDGE_PT);
arrowH(ax, 27, 33, 17.5, '', INK, LABEL, EDGE_PT);
elbow(ax, [64 70 70 76], [33.5 33.5 29 29], RED);
elbow(ax, [64 70 70 76], [17.5 17.5 22 22], RED);
text(ax, 69.4, 31.4, 'veto', 'HorizontalAlignment', 'right', 'FontName', FONT, ...
    'FontSize', EDGE_PT, 'FontWeight', 'bold', 'Color', RED);
text(ax, 69.4, 19.6, 'veto', 'HorizontalAlignment', 'right', 'FontName', FONT, ...
    'FontSize', EDGE_PT, 'FontWeight', 'bold', 'Color', RED);

% A hazard stop drops straight out of the controller into the explorer.
arrowV(ax, 95, 42, 36, 'hazard stop', 'right', INK, LABEL, EDGE_PT);

% ── Band 4: actuation. Both the speed policy and the explorer write cmd_vel.
drawbox(ax, 113, 20, 23, 14, actFill, actEdge, 'cmd\_vel bridge',  'Twist -> RoverCommand');
drawbox(ax, 113,  1, 23, 14, actFill, actEdge, 'Six wheel motors', 'PCA9685 PWM HAT');
elbow(ax, [108 124 124 124], [54 54 54 34], INK);
text(ax, 116, 56.4, '/exomy/cmd\_vel', 'HorizontalAlignment', 'center', ...
    'FontName', FONT, 'FontSize', EDGE_PT, 'Color', LABEL);
arrowH(ax, 102, 113, 25.5, '', INK, LABEL, EDGE_PT);
text(ax, 107.5, 31.2, 'overrides', 'HorizontalAlignment', 'center', 'FontName', FONT, ...
    'FontSize', EDGE_PT, 'Color', LABEL);
text(ax, 107.5, 28.0, 'cmd\_vel', 'HorizontalAlignment', 'center', 'FontName', FONT, ...
    'FontSize', EDGE_PT, 'Color', LABEL);
arrowV(ax, 124, 20, 15, 'hardware only', 'left', INK, LABEL, EDGE_PT);

% ── Key.
rectangle(ax, 'Position', [3 0.5 102 8.6], 'Curvature', 0.10, ...
    'FaceColor', [0.98 0.98 0.98], 'EdgeColor', [0.65 0.65 0.65], 'LineWidth', 1.0);
sw = {sensorFill, sensorEdge, 'sensor'; percFill, percEdge, 'foundation model'; ...
      ctrlFill, ctrlEdge, 'decision'; vetoFill, vetoEdge, 'veto only'; ...
      actFill, actEdge, 'actuation'};
xk = 6;
for k = 1:5
    rectangle(ax, 'Position', [xk 6.0 3.4 2.6], 'FaceColor', sw{k,1}, ...
        'EdgeColor', sw{k,2}, 'LineWidth', 1.3);
    text(ax, xk + 4.4, 7.3, sw{k,3}, 'FontName', FONT, 'FontSize', KEY_PT, ...
        'VerticalAlignment', 'middle', 'Color', INK);
    xk = xk + 20;
end
plot(ax, [6 11], [3.0 3.0], '-', 'Color', RED, 'LineWidth', 2.0);
text(ax, 12.6, 3.0, ['red: a veto. LiDAR and IMU can refuse a heading, but neither ' ...
    'proposes a terrain class nor commands a manoeuvre of its own.'], ...
    'FontName', FONT, 'FontSize', KEY_PT, 'VerticalAlignment', 'middle', 'Color', RED);

exportgraphics(fig, fullfile(outDir, 'fm_pipeline.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outDir, 'fm_pipeline.pdf'));
fprintf('  wrote fm_pipeline\n');

%% ── Figure 3.6: reactive-explorer state machine ─────────────────────────
fprintf('State machine diagram...\n');
fig2 = figure('Position', [100 100 1400 1000], 'Color', 'w', 'Visible', 'off');
ax2 = axes(fig2, 'Position', [0 0 1 1]); hold(ax2, 'on');
axis(ax2, [-2 134 0 100]); axis(ax2, 'off');

text(ax2, 0, 97.6, 'Reactive-explorer state machine, as shipped', ...
    'FontName', FONT, 'FontSize', TITLE_PT, 'FontWeight', 'bold', 'Color', INK);

drawbox(ax2, 14, 74, 42, 14, sensorFill, sensorEdge, 'STARTUP\_CHECK',    'settle, verify odom and IMU');
drawbox(ax2, 76, 74, 42, 14, sensorFill, sensorEdge, 'STARTUP\_SWEEP',    'look around once, ~85 s');
drawbox(ax2, 14, 52, 42, 14, sensorFill, sensorEdge, 'MONITORING',        'drive, cede cmd\_vel while clear');
drawbox(ax2, 76, 52, 42, 14, sensorFill, sensorEdge, 'TURN\_TO\_HEADING', 'closed-loop turn on yaw');
drawbox(ax2, 40, 28, 50, 13, failFill,   failEdge,   'FAILSAFE',          'terminal, wheels stopped');

% The sweep runs once, on the first pass through STARTUP_CHECK only.
arrowH(ax2, 56, 76, 83.0, 'first entry only', INK, LABEL, EDGE_PT);

% Both startup states leave through the SAME check: is the rover already
% facing a heading the picker accepts? If so it drives, if not it turns
% first. Drawn as one bus rather than four separate arrows, because in the
% code it is one branch (_step_startup_check and _step_startup_sweep both
% end in the same "already facing it?" test).
arrowV(ax2, 35, 74, 66, '', 'left', INK, LABEL, EDGE_PT);
lineonly(ax2, [97 97], [74 70], INK);
lineonly(ax2, [35 97], [70 70], INK);
arrowV(ax2, 97, 70, 66, '', 'right', INK, LABEL, EDGE_PT);
text(ax2, 66, 71.6, 'already facing a heading the picker accepts?', ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', EDGE_PT, ...
    'Color', LABEL);
text(ax2, 32.8, 68.2, 'yes', 'HorizontalAlignment', 'right', ...
    'FontName', FONT, 'FontSize', EDGE_PT, 'Color', LABEL);
text(ax2, 99.2, 68.2, 'no', 'HorizontalAlignment', 'left', ...
    'FontName', FONT, 'FontSize', EDGE_PT, 'Color', LABEL);

arrowH(ax2, 56, 76, 59.0, 'hazard held 3 s', INK, LABEL, EDGE_PT);

% A completed turn returns to STARTUP_CHECK, NOT to MONITORING. The rover
% has just been rotating, so the settle window and the re-scan both have to
% run again before it commits to driving: accelerometer-derived tilt is
% corrupted by the rotation that just ended, and the corridor ahead is a
% different piece of world than the one scanned before the turn.
lineonly(ax2, [118 130], [59 59], INK);
lineonly(ax2, [130 130], [59 94], INK);
lineonly(ax2, [130 35],  [94 94], INK);
arrowV(ax2, 35, 94, 88, '', 'right', INK, LABEL, EDGE_PT);
text(ax2, 84, 95.8, 'turn complete: settle and re-scan before driving', ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', EDGE_PT, ...
    'Color', LABEL);

% ── Terminal transitions, collected onto one bus. The drops carry no
%    arrowheads of their own: a head mid-route reads as a destination, and
%    the only destination here is FAILSAFE.
lineonly(ax2, [14 6],     [81 81], RED);
lineonly(ax2, [6 6],      [81 45], RED);
lineonly(ax2, [118 124],  [81 81], RED);
lineonly(ax2, [124 124],  [81 45], RED);
lineonly(ax2, [24 24],    [52 45], RED);
lineonly(ax2, [97 97],    [52 45], RED);
lineonly(ax2, [6 124],    [45 45], RED);
arrowV(ax2, 65, 45, 41, '', 'right', RED, RED, EDGE_PT);
% Numbered left to right as they sit on the page, not in state order: a
% reader scans the bus, they do not know the state order in advance.
tag(ax2, '1',   2.4, 49.4, RED);   % STARTUP_CHECK
tag(ax2, '2',  20.4, 49.4, RED);   % MONITORING
tag(ax2, '3',  93.4, 49.4, RED);   % TURN_TO_HEADING
tag(ax2, '4', 120.4, 49.4, RED);   % STARTUP_SWEEP

% ── Note: the trigger names, the IMU preemption, and the absent reverse state.
rectangle(ax2, 'Position', [-1 1 133 25], 'Curvature', 0.06, ...
    'FaceColor', [0.995 0.955 0.955], 'EdgeColor', RED, 'LineWidth', 1.2, ...
    'LineStyle', '--');
text(ax2, 65, 22.4, 'IMU over-tilt preempts every state except STARTUP\_CHECK', ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', BOX_PT, ...
    'FontWeight', 'bold', 'Color', [0.56 0.10 0.10]);
text(ax2, 65, 18.8, ['The rover stops, re-scans, and turns towards a clear heading. ' ...
    'It never commands a manoeuvre of its own, and there is no reverse state.'], ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', SUB_PT, ...
    'Color', [0.30 0.14 0.14]);
text(ax2, 65, 15.6, 'A stale LiDAR scan (lidar\_stale) ends the mission from any state.', ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', SUB_PT, ...
    'Color', [0.30 0.14 0.14]);
text(ax2, 65, 12.0, 'Red transitions end the mission:', 'HorizontalAlignment', 'center', ...
    'FontName', FONT, 'FontSize', SUB_PT, 'FontWeight', 'bold', 'Color', RED);
text(ax2, 65, 8.6, ['1  STARTUP\_CHECK: imu\_not\_ready, over\_tilted, boxed\_in, ' ...
    'terrain\_confirm\_timeout, terrain\_rejected\_everywhere, terrain\_search\_timeout'], ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', SUB_PT, 'Color', RED);
text(ax2, 65, 5.4, ['2  MONITORING: boxed\_in, no\_room\_to\_turn        ' ...
    '3  TURN\_TO\_HEADING: turn\_timeout, odom\_stale'], ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', SUB_PT, 'Color', RED);
text(ax2, 65, 2.2, '4  STARTUP\_SWEEP: sweep\_turn\_timeout, no\_room\_to\_turn, boxed\_in', ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', SUB_PT, 'Color', RED);

exportgraphics(fig2, fullfile(outDir, 'reactive_fsm.png'), 'Resolution', 300);
exportgraphics(fig2, fullfile(outDir, 'reactive_fsm.pdf'));
fprintf('  wrote reactive_fsm\n');

%% ── Poster variant: same machine, no trigger footnote ───────────────────
% The five-line list of failsafe trigger names is unreadable at poster size
% and is detail a poster reader does not need; the numbered markers go with
% it, since they exist only to index that list. The report keeps both.
fprintf('State machine diagram, poster variant...\n');
% Every text and every rectangle goes, not just the dashed note box: the
% numbered tag markers are rectangles with Curvature 1 and survived a
% narrower delete, leaving four empty red circles on the poster.
delete(findobj(ax2, 'Type', 'text'));
delete(findobj(ax2, 'Type', 'rectangle'));
axis(ax2, [-2 134 18 100]);

drawbox(ax2, 14, 74, 42, 14, sensorFill, sensorEdge, 'STARTUP\_CHECK',    'settle, verify odom and IMU');
drawbox(ax2, 76, 74, 42, 14, sensorFill, sensorEdge, 'STARTUP\_SWEEP',    'look around once, ~85 s');
drawbox(ax2, 14, 52, 42, 14, sensorFill, sensorEdge, 'MONITORING',        'drive, cede cmd\_vel while clear');
drawbox(ax2, 76, 52, 42, 14, sensorFill, sensorEdge, 'TURN\_TO\_HEADING', 'closed-loop turn on yaw');
drawbox(ax2, 40, 28, 50, 13, failFill,   failEdge,   'FAILSAFE',          'terminal, wheels stopped');

text(ax2, 66, 71.6, 'already facing a heading the picker accepts?', ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', EDGE_PT, ...
    'Color', LABEL);
text(ax2, 32.8, 68.2, 'yes', 'HorizontalAlignment', 'right', ...
    'FontName', FONT, 'FontSize', EDGE_PT, 'Color', LABEL);
text(ax2, 99.2, 68.2, 'no', 'HorizontalAlignment', 'left', ...
    'FontName', FONT, 'FontSize', EDGE_PT, 'Color', LABEL);
text(ax2, (56+76)/2, 85.6, 'first entry only', 'HorizontalAlignment', 'center', ...
    'FontName', FONT, 'FontSize', EDGE_PT, 'Color', LABEL);
text(ax2, (56+76)/2, 61.6, 'hazard held 3 s', 'HorizontalAlignment', 'center', ...
    'FontName', FONT, 'FontSize', EDGE_PT, 'Color', LABEL);
text(ax2, 84, 95.8, 'turn complete: settle and re-scan before driving', ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', EDGE_PT, ...
    'Color', LABEL);
text(ax2, 66, 22.5, 'Red: every route that ends the mission. The rover stops; it never guesses.', ...
    'HorizontalAlignment', 'center', 'FontName', FONT, 'FontSize', SUB_PT, ...
    'FontWeight', 'bold', 'Color', RED);

fig2.Position = [100 100 1400 760];
exportgraphics(fig2, fullfile(outDir, 'reactive_fsm_poster.png'), 'Resolution', 300);
exportgraphics(fig2, fullfile(outDir, 'reactive_fsm_poster.pdf'));
fprintf('  wrote reactive_fsm_poster\n');

fprintf('done\n');

%% ── helpers (must stay at the very end: local functions mid-script make
%%    matlab -batch hang until timeout with no error) ─────────────────────

function drawbox(ax, x, y, w, h, fill, edge, titleStr, subStr)
    rectangle(ax, 'Position', [x y w h], 'Curvature', 0.10, ...
        'FaceColor', fill, 'EdgeColor', edge, 'LineWidth', 1.6);
    if isempty(titleStr); return; end
    if isempty(subStr)
        text(ax, x+w/2, y+h/2, titleStr, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontName', 'Arial', ...
            'FontSize', 14.5, 'FontWeight', 'bold', 'Color', [0.12 0.12 0.12]);
    else
        text(ax, x+w/2, y+h*0.66, titleStr, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontName', 'Arial', ...
            'FontSize', 14.5, 'FontWeight', 'bold', 'Color', [0.12 0.12 0.12]);
        text(ax, x+w/2, y+h*0.27, subStr, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontName', 'Arial', ...
            'FontSize', 12.5, 'Color', [0.26 0.26 0.26]);
    end
end

function head(ax, x, y, dirVec, col)
    u = dirVec / norm(dirVec); n = [-u(2) u(1)]; s = 1.7;
    p1 = [x y]; p2 = p1 - u*s*1.8 + n*s*0.78; p3 = p1 - u*s*1.8 - n*s*0.78;
    patch(ax, 'XData', [p1(1) p2(1) p3(1)], 'YData', [p1(2) p2(2) p3(2)], ...
        'FaceColor', col, 'EdgeColor', 'none');
end

function arrowH(ax, x1, x2, y, lab, col, labCol, pt)
    plot(ax, [x1 x2], [y y], '-', 'Color', col, 'LineWidth', 1.7);
    head(ax, x2, y, [sign(x2-x1) 0], col);
    if ~isempty(lab)
        % Above the line, never on it.
        text(ax, (x1+x2)/2, y + 2.6, lab, 'HorizontalAlignment', 'center', ...
            'FontName', 'Arial', 'FontSize', pt, 'Color', labCol);
    end
end

function arrowV(ax, x, y1, y2, lab, side, col, labCol, pt)
    plot(ax, [x x], [y1 y2], '-', 'Color', col, 'LineWidth', 1.7);
    head(ax, x, y2, [0 sign(y2-y1)], col);
    if isempty(lab); return; end
    % Beside the line, never on it.
    if strcmp(side, 'left')
        text(ax, x - 2.2, (y1+y2)/2, lab, 'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'middle', 'FontName', 'Arial', ...
            'FontSize', pt, 'Color', labCol);
    else
        text(ax, x + 2.2, (y1+y2)/2, lab, 'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'middle', 'FontName', 'Arial', ...
            'FontSize', pt, 'Color', labCol);
    end
end

function lineonly(ax, xs, ys, col)
    % A connector segment with NO arrowhead. Used for the legs that feed a
    % shared bus: an arrowhead part-way along a route reads as "this is where
    % the transition ends", which is wrong for a leg that is still in transit.
    plot(ax, xs, ys, '-', 'Color', col, 'LineWidth', 1.7);
end

function elbow(ax, xs, ys, col)
    plot(ax, xs, ys, '-', 'Color', col, 'LineWidth', 1.7);
    d = [xs(end)-xs(end-1), ys(end)-ys(end-1)];
    if norm(d) == 0; d = [xs(end)-xs(end-2), ys(end)-ys(end-2)]; end
    head(ax, xs(end), ys(end), d, col);
end

function tag(ax, lab, x, y, col)
    % A numbered marker on a terminal-transition drop; the trigger names it
    % stands for are listed in the note at the foot of the figure.
    rectangle(ax, 'Position', [x-2.1 y-2.1 4.2 4.2], 'Curvature', 1, ...
        'FaceColor', 'w', 'EdgeColor', col, 'LineWidth', 1.4);
    text(ax, x, y, lab, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'FontName', 'Arial', ...
        'FontSize', 12.5, 'FontWeight', 'bold', 'Color', col);
end
