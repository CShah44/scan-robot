clc; clear; close all;

%% ------------------------------------
% Load environment
% -------------------------------------
[clients, obstacles, mapW, mapH] = environment();

%% ------------------------------------
% NSGA-II parameters
% -------------------------------------
nRouters = 5;
nvars = 2 * nRouters;

lb = ones(1, nvars);
ub = repmat([mapW mapH], 1, nRouters);

options = optimoptions('gamultiobj', ...
    'FunctionTolerance', 0, ...
    'PopulationSize', 50, ...
    'MaxGenerations', 500, ...
    'CrossoverFraction', 0.9, ...   % SBX-style behavior
    'CrossoverFcn', {@crossovderscattered}, ...
    'Display', 'iter');

%% ------------------------------------
% Run NSGA-II
% -------------------------------------
[xPareto, fPareto] = gamultiobj( ...
    @(x) wifiObjectives(x, clients, obstacles), ...
    nvars, [], [], [], [], lb, ub, options);

%Size of Pareto
size(fPareto)

%% ------------------------------------
% Plot Pareto front
% -------------------------------------
figure;
scatter3( ...
    -fPareto(:,1), ...   % Coverage
    -fPareto(:,2), ...   % RSSI
     fPareto(:,3), ...   % Overlap
    40, fPareto(:,3), 'filled');

xlabel('Coverage Ratio');
ylabel('Mean RSSI (dBm)');
zlabel('Overlap');
title('3D Pareto Front (Coverage vs RSSI vs Overlap)');
grid on;
view(45,30);


%% ------------------------------------
% Select one solution (max coverage)
% -------------------------------------
[~, idx] = max(-fPareto(:,1));
bestX = xPareto(idx,:);
routers = reshape(bestX, [2 nRouters])';

%% ------------------------------------
% Visualize best layout
% -------------------------------------
figure('Color','w'); hold on;
axis equal;
xlim([0 mapW]);
ylim([0 mapH]);
grid on;

% Obstacles
for i = 1:size(obstacles,1)
    rectangle('Position', obstacles(i,:), ...
        'FaceColor', [0.6 0.6 0.6], ...
        'EdgeColor', 'k', ...
        'LineWidth', 1.5);
end

% Clients
scatter(clients(:,1), clients(:,2), 40, 'b', 'filled');

% Routers
scatter(routers(:,1), routers(:,2), 160, 'r', 'p', ...
    'filled', 'MarkerEdgeColor','w');

% Coverage circles
for i = 1:nRouters
    viscircles(routers(i,:), 30, ...
        'Color','r','LineStyle','--','LineWidth',0.7);
end

legend('Obstacle','Clients','Routers');
title('Optimal Router Placement (NSGA-II)');
xlabel('X');
ylabel('Y');

hold off;

%% ------------------------------------
% Final performance metrics (AREA overlap)
%% ------------------------------------
RSSImin = -70;
gridRes = 3;

covered = 0;
totalRSSI = 0;

xGrid = 1:gridRes:mapW;
yGrid = 1:gridRes:mapH;

overlapCells = 0;

for c = 1:size(clients,1)
    rssi_vals = zeros(nRouters,1);
    for r = 1:nRouters
        rssi_vals(r) = computeRSSI(routers(r,:), clients(c,:), obstacles);
    end
    bestRSSI = max(rssi_vals);
    if bestRSSI >= RSSImin
        covered = covered + 1;
        totalRSSI = totalRSSI + bestRSSI;
    end
end

for ix = 1:length(xGrid)
    for iy = 1:length(yGrid)
        p = [xGrid(ix), yGrid(iy)];
        count = 0;
        for r = 1:nRouters
            if computeRSSI(routers(r,:), p, obstacles) >= RSSImin
                count = count + 1;
            end
        end
        if count > 1
            overlapCells = overlapCells + 1;
        end
    end
end

overlapArea = overlapCells * gridRes^2;

avgRSSI = totalRSSI / max(covered,1);
coverageRatio = covered / size(clients,1);

%% ------------------------------------
% WiFi RSSI Heatmap Visualization
%% ------------------------------------
gridRes = 1;          % resolution for visualization
RSSImin = -80;

xGrid = 1:gridRes:mapW;
yGrid = 1:gridRes:mapH;

RSSI_map = zeros(length(yGrid), length(xGrid));

for ix = 1:length(xGrid)
    for iy = 1:length(yGrid)
        p = [xGrid(ix), yGrid(iy)];
        bestRSSI = -Inf;

        for r = 1:nRouters
            rssi = computeRSSI(routers(r,:), p, obstacles);
            bestRSSI = max(bestRSSI, rssi);
        end

        RSSI_map(iy, ix) = bestRSSI;
    end
end

%% ------------------------------------
% Plot heatmap
%% ------------------------------------
figure('Color','w'); hold on;

imagesc(xGrid, yGrid, RSSI_map);
set(gca, 'YDir','normal');
colormap(jet);
colorbar;
caxis([-90 -20]);   % clamp RSSI values

xlim([0 mapW]);
ylim([0 mapH]);
axis equal;
hold on;

% Obstacles
for i = 1:size(obstacles,1)
    rectangle('Position', obstacles(i,:), ...
        'FaceColor', [0.4 0.4 0.4], ...
        'EdgeColor','k', ...
        'LineWidth',1.2);
end

% Routers
scatter(routers(:,1), routers(:,2), ...
    160, 'w', 'p', 'filled', 'MarkerEdgeColor','k');

% Clients
scatter(clients(:,1), clients(:,2), ...
    40, 'k', 'filled');

title('WiFi RSSI Heatmap (Max RSSI from Routers)');
xlabel('X');
ylabel('Y');

hold off;


fprintf('\n===== FINAL NSGA-II SOLUTION (AREA OVERLAP) =====\n');
fprintf('Clients covered          : %d / %d (%.2f %%)\n', ...
        covered, size(clients,1), coverageRatio*100);
fprintf('Average RSSI (covered)   : %.2f dBm\n', avgRSSI);
fprintf('Overlap area             : %.2f m^2\n', overlapArea);
fprintf('================================================\n\n');
