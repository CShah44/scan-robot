clc; clear; close all;

[clients, obstacles, mapW, mapH] = environment(42);

figure('Color','w');
hold on;
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
scatter(clients(:,1), clients(:,2), ...
        40, 'b', 'filled');

legend('Clients','Location','northwest');
title('WiFi Router Placement Layout (100x100)');
xlabel('X');
ylabel('Y');

hold off;