function f = wifiObjectives(x, clients, obstacles)
    gridRes = 3;
    mapW = 100;
    mapH = 100;

    nRouters = length(x)/2;
    routers = reshape(x, [2 nRouters])';

    RSSImin = -80;

    covered = 0;
    totalRSSI = 0;

    for c = 1:size(clients,1)

        rssi_vals = zeros(nRouters,1);

        for r = 1:nRouters
            rssi_vals(r) = computeRSSI(routers(r,:), clients(c,:), obstacles);
        end

        % Coverage and RSSI
        bestRSSI = max(rssi_vals);
        if bestRSSI >= RSSImin
            covered = covered + 1;
            totalRSSI = totalRSSI + bestRSSI;
        

        end
    end

    coverageRatio = covered / size(clients,1);
    
    % Calculates mean rssi across covered clients
    %meanRSSI = (covered > 0) * (totalRSSI / max(covered,1)) + ...
     %          (covered == 0) * (-100);
     meanRSSI = totalRSSI / size(clients,1);

    %% ------------------------------------
    % Objective 3: Overlap AREA
    %% ------------------------------------
    xGrid = 1:gridRes:mapW;
    yGrid = 1:gridRes:mapH;

    overlapCells = 0;
    totalCells = numel(xGrid) * numel(yGrid);

    for ix = 1:length(xGrid)
        for iy = 1:length(yGrid)

            p = [xGrid(ix), yGrid(iy)];
            count = 0;

            for r = 1:nRouters
                rssi = computeRSSI(routers(r,:), p, obstacles);
                if rssi >= RSSImin
                    count = count + 1;
                end
            end

            if count > 1
                overlapCells = overlapCells + 1;
            end
        end
    end

    cellArea = gridRes^2;
    overlapArea = overlapCells * cellArea;
    % Objective vector (MINIMIZATION)
    f(1) = -coverageRatio;   % maximize coverage
    f(2) = -meanRSSI;        % maximize RSSI
    f(3) = overlapArea;      % minimize overlap area
end
