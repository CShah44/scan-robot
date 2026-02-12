function [clients, obstacles, mapW, mapH] = environment(seed)

if nargin < 1 || isempty(seed)
    seed = 42;
end

rng(seed);

mapW = 100;
mapH = 100;

obstacles = [
    20 20 60 5;
    20 20 5 40;
    75 20 5 40;
    20 75 60 5;
    40 40 20 20
];

% Python: np.random.randint(0, GRID_SIZE)
clients = randi([0 mapW-1], 20, 2);

end