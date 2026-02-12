clc; clear; close all;

% Global parameters for custom operators
global ETA_C ETA_M MUTATION_RATE GRID_SIZE;
ETA_C = 20;
ETA_M = 20;
MUTATION_RATE = 0.1;
GRID_SIZE = 100;

rng(42);

%% ------------------------------------
% Load environment
% -------------------------------------
[clients, obstacles, mapW, mapH] = environment(42);

%% ------------------------------------
% NSGA-II parameters (Python-aligned)
% -------------------------------------
nRouters = 5;
nvars = 2 * nRouters;

lb = zeros(1, nvars);
ub = repmat([mapW mapH], 1, nRouters);

options = optimoptions('gamultiobj', ...
    'FunctionTolerance', 0, ...
    'PopulationSize', 50, ...
    'MaxGenerations', 100, ...
    'CrossoverFraction', 1.0, ...
    'CrossoverFcn', @sbxCrossover, ...
    'MutationFcn', @polynomialMutation, ...
    'ParetoFraction', 1.0, ...
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

legend('Obstacle','Clients','Routers');
title('Optimal Router Placement (NSGA-II)');
xlabel('X');
ylabel('Y');

hold off;

%% ------------------------------------
% Final performance metrics (Python-aligned)
%% ------------------------------------
RSSIminCoverage = -80;
RSSIminOverlap  = -70;
gridRes = 2;

covered = 0;
totalRSSI = 0;

for c = 1:size(clients,1)
    rssi_vals = zeros(nRouters,1);
    for r = 1:nRouters
        rssi_vals(r) = computeRSSI(routers(r,:), clients(c,:), obstacles);
    end
    bestRSSI = max(rssi_vals);
    if bestRSSI > RSSIminCoverage
        covered = covered + 1;
    end
    totalRSSI = totalRSSI + bestRSSI;
end

xGrid = 0:gridRes:(mapW-gridRes);
yGrid = 0:gridRes:(mapH-gridRes);

overlapCells = 0;

for ix = 1:length(xGrid)
    for iy = 1:length(yGrid)
        p = [xGrid(ix), yGrid(iy)];
        count = 0;
        for r = 1:nRouters
            if computeRSSI(routers(r,:), p, obstacles) > RSSIminOverlap
                count = count + 1;
            end
        end
        if count > 1
            overlapCells = overlapCells + 1;
        end
    end
end

totalCells = length(xGrid) * length(yGrid);
overlapArea = (overlapCells / totalCells) * (mapW * mapH);

avgRSSI = totalRSSI / size(clients,1);
coverageRatio = covered / size(clients,1);

%% ------------------------------------
% WiFi RSSI Heatmap Visualization
%% ------------------------------------
gridRes = 0.5;
xGrid = 0:gridRes:(mapW-gridRes);
yGrid = 0:gridRes:(mapH-gridRes);

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
caxis([-90 -40]);

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

fprintf('\n===== FINAL NSGA-II SOLUTION (PYTHON-ALIGNED) =====\n');
fprintf('Clients covered          : %d / %d (%.2f %%)\n', ...
        covered, size(clients,1), coverageRatio*100);
fprintf('Average RSSI (all clients): %.2f dBm\n', avgRSSI);
fprintf('Overlap area             : %.2f m^2\n', overlapArea);
fprintf('==================================================\n\n');

%% ------------------------------------
% Local Crossover + Mutation
%% ------------------------------------
function xoverKids = sbxCrossover(parents, ~, nvars, ~, ~, thisPopulation)
    global ETA_C GRID_SIZE
    nKids = floor(length(parents)/2);
    xoverKids = zeros(nKids, nvars);

    for k = 1:nKids
        p1 = thisPopulation(parents(2*k-1), :);
        p2 = thisPopulation(parents(2*k), :);
        child = zeros(1, nvars);

        for i = 1:nvars
            val1 = p1(i);
            val2 = p2(i);

            if rand <= 0.5
                if abs(val1 - val2) > 1e-6
                    u = rand;
                    if u <= 0.5
                        beta = (2*u)^(1/(ETA_C + 1));
                    else
                        beta = (1/(2*(1-u)))^(1/(ETA_C + 1));
                    end
                    c1 = 0.5 * ((1 + beta)*val1 + (1 - beta)*val2);
                    c2 = 0.5 * ((1 - beta)*val1 + (1 + beta)*val2);
                else
                    c1 = val1; c2 = val2;
                end
            else
                c1 = val1; c2 = val2;
            end

            % gamultiobj accepts one child per pair
            child(i) = min(max(c1, 0), GRID_SIZE);
        end

        xoverKids(k, :) = child;
    end
end

function mutationChildren = polynomialMutation(parents, ~, nvars, ~, ~, ~, thisPopulation)
    global ETA_M MUTATION_RATE GRID_SIZE
    mutationChildren = thisPopulation(parents, :);

    for i = 1:size(mutationChildren, 1)
        for j = 1:nvars
            if rand < MUTATION_RATE
                val = mutationChildren(i, j);
                low = 0.0; high = GRID_SIZE;
                delta_range = high - low;

                u = rand;
                rk = (val - low) / delta_range;

                if u <= 0.5
                    delta_q = (2*u + (1 - 2*u) * (1 - rk)^(ETA_M + 1))^(1/(ETA_M + 1)) - 1;
                else
                    delta_q = 1 - (2*(1 - u) + 2*(u - 0.5) * (1 - rk)^(ETA_M + 1))^(1/(ETA_M + 1));
                end

                new_val = val + delta_q * delta_range;
                mutationChildren(i, j) = min(max(new_val, low), high);
            end
        end
    end
end