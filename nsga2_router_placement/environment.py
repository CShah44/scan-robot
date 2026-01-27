import numpy as np
from .config import *

class Environment:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.grid_size = GRID_SIZE
        # Define 20 fixed clients
        # We'll distribute them somewhat randomly but kept fixed
        self.clients = np.random.randint(0, GRID_SIZE, size=(N_CLIENTS, 2))
        
        # Define Obstacles (Rectangles: [x, y, width, height])
        # Creating a simple floor plan-like structure
        self.obstacles = [
            [20, 20, 60, 5],   # Horizontal wall
            [20, 20, 5, 40],   # Vertical wall 1
            [75, 20, 5, 40],   # Vertical wall 2
            [20, 75, 60, 5],   # Top horizontal wall
            [40, 40, 20, 20],  # Central block
        ]
        self.obstacles = np.array(self.obstacles)

    def is_obstructed(self, p1, p2):
        """
        Check if the line segment p1-p2 intersects any obstacle.
        Simple ray casting check against rectangles.
        Returns the number of walls intersected.
        """
        n_walls = 0
        for obs in self.obstacles:
            x, y, w, h = obs
            # Obstacle bounds
            ox1, oy1 = x, y
            ox2, oy2 = x + w, y + h
            
            # Line segment p1-p2
            x1, y1 = p1
            x2, y2 = p2
            
            # Check intersection with the 4 rectangle edges
            # Top: (ox1, oy2) -> (ox2, oy2)
            if self._intersect(x1, y1, x2, y2, ox1, oy2, ox2, oy2): n_walls += 1
            # Bottom: (ox1, oy1) -> (ox2, oy1)
            if self._intersect(x1, y1, x2, y2, ox1, oy1, ox2, oy1): n_walls += 1
            # Left: (ox1, oy1) -> (ox1, oy2)
            if self._intersect(x1, y1, x2, y2, ox1, oy1, ox1, oy2): n_walls += 1
            # Right: (ox2, oy1) -> (ox2, oy2)
            if self._intersect(x1, y1, x2, y2, ox2, oy1, ox2, oy2): n_walls += 1
            
            # Also check if either point is INSIDE the obstacle (highly attenuated or impossible placement)
            # For signal, if router is inside, maybe valid but attenuated? 
            # Impl says "where routers cannot be placed". We handle placement constraints separately or here?
            # For now just counting wall storage.
            
        return n_walls

    def _intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Standard line segment intersection."""
        # Denominator
        d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if d == 0: return False
        
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
        
        return 0 <= ua <= 1 and 0 <= ub <= 1

    def calculate_rssi(self, router_pos, target_pos):
        d = np.linalg.norm(router_pos - target_pos)
        if d < 1e-6: d = 1e-6 # Avoid log(0)
        
        n_walls = self.is_obstructed(router_pos, target_pos)
        
        # Log-Distance Path Loss
        # RSSI = P_tx - (L0 + 10 * gamma * log10(d) + walls * L_wall)
        path_loss = L0 + 10 * GAMMA * np.log10(d) + n_walls * OBSTACLE_ATTENUATION
        rssi = P_TX - path_loss
        return rssi

    def evaluate(self, routers):
        """
        Calculate the 3 objectives:
        f1: Maximize Coverage (Clients with RSSI > -80) -> We will MINIMIZE (-1 * Coverage)
        f2: Maximize Signal Quality (Avg RSSI) -> We will MINIMIZE (-1 * Avg RSSI)
        f3: Minimize Overlap (Area with >1 router > -70) -> We will MINIMIZE Overlap
        """
        # 1. Evaluate coverage and quality for CLIENTS
        client_rssi_values = []
        for client in self.clients:
            # RSSI from each router to this client
            rssis = [self.calculate_rssi(r, client) for r in routers]
            best_rssi = max(rssis)
            client_rssi_values.append(best_rssi)
        
        client_rssi_values = np.array(client_rssi_values)
        
        # f1: Coverage Ratio
        covered_count = np.sum(client_rssi_values > MIN_RSSI_COVERAGE)
        coverage_ratio = covered_count / N_CLIENTS
        
        # f2: Average RSSI (Signal Quality)
        avg_rssi = np.mean(client_rssi_values)
        
        # f3: Overlap
        # We need to check the whole grid.
        # Vectorized check might be hard with 'is_obstructed' having loops using simple method.
        # For performance, maybe we sample the grid or use a simplified model for the grid map?
        # The prompt asks for "Total grid area", implying 100x100 resolution.
        # Doing 10000 ray casts * 5 routers = 50k ray casts per individual is SLOW in Python.
        # Optimization: Only check vicinity? Or use a pre-calculated visibility map?
        # Since obstacles are static, we can pre-calculate distance/walls map for all grid points?
        # But routers move. 
        # Let's try sampling or a coarser grid for overlap calculation to speed it up, 
        # say 10x10 blocks (100 points total) or 50x50.
        # Or just do the full grid if we can vectorize calculate_rssi.
        
        # Let's implement full grid check but optimize later if slow.
        # Actually, for 50 gen * 50 pop = 2500 evals. 
        # If each takes 1s, that's 40 mins. Too slow.
        # We need a faster overlap check.
        # Let's assume for overlap we simply check distance without walls first? 
        # "Strict NSGA-II ... Include a basic Ray-Casting ... for walls".
        # Okay, let's optimize `is_obstructed`.
        
        # For this step, I will implement a basic version.
        # To make it fast, I'll sample the grid with step 2 (50x50 = 2500 points).
        
        step = 2
        grid_x, grid_y = np.mgrid[0:GRID_SIZE:step, 0:GRID_SIZE:step]
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        
        overlap_count = 0
        # Calculate RSSI for all routers to all grid points
        # This is the bottleneck.
        # Simplified: Just distance based P_TX - (L0 + 10*gamma*log(d)) > -70?
        # Ignore walls for overlap to speed up?
        # The prompt implies strict "Include ... check to count wall intersections" for RSSI model. 
        # It doesn't explicitly say "ignore walls for overlap".
        
        # Strategy: Precompute Wall Map? No, routers move.
        # We'll use a coarser grid for overlap (Step=5 -> 20x20 = 400 points) to keep it sane.
        
        return -coverage_ratio, -avg_rssi, 0 # Placeholder for overlap until optimization
        
    def is_obstructed_vectorized(self, p1, points):
        """
        Vectorized check for obstructions between p1 (1, 2) and points (N, 2).
        Returns number of walls (N,).
        """
        p1 = np.array(p1)
        x1, y1 = p1
        
        points = np.atleast_2d(points)
        x2, y2 = points[:, 0], points[:, 1]
        
        # print(f"DEBUG: points.shape={points.shape}, x2.shape={x2.shape}")
        
        n_intersections = np.zeros(len(points), dtype=int)
        
        # Obstacles: (M, 4) -> x, y, w, h
        # We need to check intersection with 4 lines for each obstacle
        
        for obs in self.obstacles:
            ox, oy, w, h = obs
            # 4 segments
            segments = [
                (ox, oy, ox+w, oy),     # Bottom
                (ox, oy+h, ox+w, oy+h), # Top
                (ox, oy, ox, oy+h),     # Left
                (ox+w, oy, ox+w, oy+h)  # Right
            ]
            
            for sx1, sy1, sx2, sy2 in segments:
                d = (sy2 - sy1) * (x2 - x1) - (sx2 - sx1) * (y2 - y1)
                
                mask_d_nonzero = np.abs(d) > 1e-9
                
                ua = np.zeros_like(d)
                ub = np.zeros_like(d)
                
                num_ua = ((sx2 - sx1) * (y1 - sy1) - (sy2 - sy1) * (x1 - sx1))
                num_ub = ((x2 - x1) * (y1 - sy1) - (y2 - y1) * (x1 - sx1))
                
                # num_ua is scalar (independent of x2,y2), num_ub is vector
                
                ua[mask_d_nonzero] = num_ua / d[mask_d_nonzero]
                ub[mask_d_nonzero] = num_ub[mask_d_nonzero] / d[mask_d_nonzero]
                
                intersect = (0 <= ua) & (ua <= 1) & (0 <= ub) & (ub <= 1)
                n_intersections += intersect.astype(int)

                
        return n_intersections

    def calculate_rssi_vectorized(self, router_pos, target_points):
        # router_pos: (2,)
        # target_points: (N, 2)
        
        d = np.linalg.norm(target_points - router_pos, axis=1)
        d = np.maximum(d, 1e-6)
        
        n_walls = self.is_obstructed_vectorized(router_pos, target_points)
        
        path_loss = L0 + 10 * GAMMA * np.log10(d) + n_walls * OBSTACLE_ATTENUATION
        rssi = P_TX - path_loss
        return rssi

    def evaluate_full(self, routers):
        # Fully vectorized evaluation
        
        # 1. Client Coverage & Quality
        # Use vectorized RSSI for clients too
        client_rssi_matrix = np.zeros((N_CLIENTS, N_ROUTERS))
        for i, r in enumerate(routers):
            client_rssi_matrix[:, i] = self.calculate_rssi_vectorized(r, self.clients)
            
        max_rssis = np.max(client_rssi_matrix, axis=1)
        
        coverage = np.sum(max_rssis > MIN_RSSI_COVERAGE) / N_CLIENTS
        quality = np.mean(max_rssis)
        
        # 2. Overlap
        if not hasattr(self, 'grid_points'):
            step = 2 
            x_range = np.arange(0, GRID_SIZE, step)
            y_range = np.arange(0, GRID_SIZE, step)
            gx, gy = np.meshgrid(x_range, y_range)
            self.grid_points = np.vstack([gx.ravel(), gy.ravel()]).T 
            self.total_grid_area = GRID_SIZE * GRID_SIZE

        # Filter points by distance to ANY router (Rough filter)
        # Using KDTree or just distance matrix? Distance matrix (2500, 5) is tiny.
        
        # Calculate full RSSI map?
        # For each router, calc RSSI to all grid points.
        # This is 5 calls to calculate_rssi_vectorized(r, grid_points)
        # grid_points = 2500. 5 calls.
        # is_obstructed_vectorized loop 5 obstacles * 4 segments = 20 loops. 2500 ops each.
        # Total ops ~ 2500 * 20 * 5 = 250,000 ops "vectorized". Very fast.
        
        grid_rssi = np.zeros((len(self.grid_points), N_ROUTERS))
        
        for i, r in enumerate(routers):
            grid_rssi[:, i] = self.calculate_rssi_vectorized(r, self.grid_points)
            
        # Count overlaps
        # For each point, how many routers > MIN_RSSI_OVERLAP
        overlaps = np.sum(grid_rssi > MIN_RSSI_OVERLAP, axis=1)
        overlap_points = np.sum(overlaps > 1)
        
        overlap_area = (overlap_points / len(self.grid_points)) * self.total_grid_area
        
        return -coverage, -quality, overlap_area

