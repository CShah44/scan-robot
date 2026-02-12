% Line-segment intersection
function intersect = segmentsIntersect(p1, p2, p3, p4)

    x1 = p1(1); y1 = p1(2);
    x2 = p2(1); y2 = p2(2);
    x3 = p3(1); y3 = p3(2);
    x4 = p4(1); y4 = p4(2);

    d = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1);

    if abs(d) < 1e-9
        intersect = false; % parallel
        return;
    end

    ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / d;
    ub = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / d;

    intersect = (ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1);
end