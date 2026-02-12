% Main RSSI function
function rssi = computeRSSI(router, client, obstacles)

    % Parameters (from paper/image)
    Ptx   = 20;     % dBm
    L0    = 40;     % dB
    gamma = 3.0;
    alpha = 15;     % dB per wall

    % Distance
    d = norm(router - client);
    if d < 1
        d = 1;
    end

    % Count intersected walls
    N_walls = countWalls(router, client, obstacles);

    % Path loss
    PL = L0 + 10*gamma*log10(d) + N_walls * alpha;

    % RSSI
    rssi = Ptx - PL;
end
