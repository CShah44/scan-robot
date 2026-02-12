% Count number of wall-edge intersections (counts every edge)
function N = countWalls(p1, p2, obstacles)

    N = 0;

    for i = 1:size(obstacles,1)

        x = obstacles(i,1);
        y = obstacles(i,2);
        w = obstacles(i,3);
        h = obstacles(i,4);

        % Rectangle edges as line segments
        edges = [
            x   y     x+w y;     % bottom
            x+w y     x+w y+h;   % right
            x+w y+h   x   y+h;   % top
            x   y+h   x   y      % left
        ];

        for e = 1:4
            if segmentsIntersect(p1, p2, edges(e,1:2), edges(e,3:4))
                N = N + 1;
            end
        end
    end
end