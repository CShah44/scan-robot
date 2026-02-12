function f = wifiObjectives(x, clients, obstacles)
    gridRes = 2;
    mapW = 100;
    mapH = 100;

    nRouters = length(x)/2;
    routers = reshape(x, [2 nRouters])';

    RSSIminCoverage = -80;
    RSSIminOverlap = -70;

    % Coverage and Quality (mean of ALL clients)
    maxRSSIs = zeros(size(clients,1), 1);
    for c = 1:size(clients,1)
        rssi_vals = zeros(nRouters,1);
        for r = 1:nRouters
            rssi_vals(r) = computeRSSI(routers(r,:), clients(c,:), obstacles);
        end
        maxRSSIs(c) = max(rssi_vals);
    end

    covered = sum(maxRSSIs > RSSIminCoverage);
    coverageRatio = covered / size(clients,1);
    meanRSSI = mean(maxRSSIs);

    % Overlap area (Python-style scaling)
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

    f(1) = -coverageRatio;   % maximize coverage
    f(2) = -meanRSSI;        % maximize RSSI
    f(3) = overlapArea;      % minimize overlap area
end