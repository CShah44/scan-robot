
import numpy as np
from .config import *

class Environment:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.grid_size = GRID_SIZE
        # Define 20 fixed clients
        self.clients = np.random.randint(0, GRID_SIZE, size=(N_CLIENTS, 2))
        
        # Define Obstacles (Rectangles: [x, y, width, height])
        # Same layout as NSGA2
        self.obstacles = [
            [20, 20, 60, 5],   # Horizontal wall
            [20, 20, 5, 40],   # Vertical wall 1
            [75, 20, 5, 40],   # Vertical wall 2
            [20, 75, 60, 5],   # Top horizontal wall
            [40, 40, 20, 20],  # Central block
        ]
        self.obstacles = np.array(self.obstacles)

    def is_obstructed_vectorized(self, p1, points):
        """
        Vectorized check for obstructions between p1 (router) and points (N, 2).
        Returns number of walls (N,).
        """
        p1 = np.array(p1)
        x1, y1 = p1
        
        points = np.atleast_2d(points)
        x2, y2 = points[:, 0], points[:, 1]
        
        n_intersections = np.zeros(len(points), dtype=int)
        
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
                # Denominator for line intersection
                d = (sy2 - sy1) * (x2 - x1) - (sx2 - sx1) * (y2 - y1)
                
                mask_d_nonzero = np.abs(d) > 1e-9
                
                ua = np.zeros_like(d)
                ub = np.zeros_like(d)
                
                num_ua = ((sx2 - sx1) * (y1 - sy1) - (sy2 - sy1) * (x1 - sx1))
                num_ub = ((x2 - x1) * (y1 - sy1) - (y2 - y1) * (x1 - sx1))
                
                ua[mask_d_nonzero] = num_ua / d[mask_d_nonzero]
                ub[mask_d_nonzero] = num_ub[mask_d_nonzero] / d[mask_d_nonzero]
                
                intersect = (0 <= ua) & (ua <= 1) & (0 <= ub) & (ub <= 1)
                n_intersections += intersect.astype(int)
                
        return n_intersections

    def calculate_rssi_vectorized(self, router_pos, target_points):
        d = np.linalg.norm(target_points - router_pos, axis=1)
        d = np.maximum(d, 1e-6)
        
        n_walls = self.is_obstructed_vectorized(router_pos, target_points)
        
        path_loss = L0 + 10 * GAMMA * np.log10(d) + n_walls * OBSTACLE_ATTENUATION
        rssi = P_TX - path_loss
        return rssi

    def evaluate(self, routers):
        """
        Evaluate objectives:
        f1: Maximize Coverage (Min -Coverage)
        f2: Maximize Avg RSSI (Min -AvgRSSI)
        f3: Minimize Overlap (Sum of max(0, overlap_count - 1))
        """
        routers = np.array(routers)
        
        # 1. Client Metrics
        client_rssi_matrix = np.zeros((N_CLIENTS, N_ROUTERS))
        for i, r in enumerate(routers):
            client_rssi_matrix[:, i] = self.calculate_rssi_vectorized(r, self.clients)
            
        max_rssis = np.max(client_rssi_matrix, axis=1)
        
        # f1: Coverage Ratio
        covered_count = np.sum(max_rssis >= MIN_RSSI_COVERAGE)
        coverage_ratio = covered_count / N_CLIENTS
        
        # f2: Avg RSSI
        avg_rssi = np.mean(max_rssis)
        
        # f3: Overlap
        # Using a coarser grid for efficiency
        if not hasattr(self, 'grid_points_overlap'):
            step = 5  # Coarser step
            gx = np.arange(0, GRID_SIZE, step)
            gy = np.arange(0, GRID_SIZE, step)
            grid_x, grid_y = np.meshgrid(gx, gy)
            self.grid_points_overlap = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        
        n_points = len(self.grid_points_overlap)
        grid_rssi = np.zeros((n_points, N_ROUTERS))
        
        for i, r in enumerate(routers):
            grid_rssi[:, i] = self.calculate_rssi_vectorized(r, self.grid_points_overlap)
            
        # Count how many routers cover each point (> MIN_RSSI_OVERLAP)
        coverage_counts = np.sum(grid_rssi >= MIN_RSSI_OVERLAP, axis=1)
        
        # Overlap metric: sum max(0, count - 1)
        overlaps = np.maximum(0, coverage_counts - 1)
        total_overlap = np.sum(overlaps)
        
        return -coverage_ratio, -avg_rssi, total_overlap
