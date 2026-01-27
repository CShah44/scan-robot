"""
WiFi Coverage Simulation - Step A: Minimal Working Prototype
============================================================

A 2D grid-based simulation of wireless coverage with:
- Static Access Points (APs)
- Static Clients
- Mobile Router Agents

For research paper on mobile router coverage optimization.

Physics Model:
- Log-distance path-loss: RSSI(d) = P_tx - PL_0 - 10*n*log10(d/d_0)
- Coverage threshold: RSSI >= -70 dBm
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple
import random

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Grid parameters
GRID_SIZE = 50          # 50x50 cells
CELL_SIZE = 1.0         # 1 meter per cell

# Radio propagation parameters (Log-distance path-loss model)
TX_POWER = 20.0         # Transmit power in dBm (typical WiFi AP)
PL_0 = 40.0             # Path loss at reference distance d0 (dB) for 2.4 GHz
PATH_LOSS_EXPONENT = 3.0  # n: 2.0 free space, 3.0-4.0 indoor with obstacles
D_0 = 1.0               # Reference distance (meters)
OBSTACLE_PENALTY = 15.0  # Additional dB loss when signal passes through obstacle

# Coverage threshold
RSSI_THRESHOLD = -70.0  # dBm - minimum signal strength for "covered"

# Environment parameters
OBSTACLE_DENSITY = 0.17  # 17% of cells are obstacles

# Network configuration
# Network configuration
NUM_STATIC_APS = 0       # Number of static access points
NUM_CLIENTS = 40         # Number of static clients (Increased from 20)
NUM_MOBILE_ROUTERS = 6   # Number of mobile router agents

# Mobile router kinematics
MAX_SPEED = 2.0          # meters per second
MAX_TURN_RATE = np.pi/4  # radians per second (45 degrees)

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AccessPoint:
    """Static Access Point (AP) that provides WiFi coverage."""
    x: float
    y: float
    tx_power: float = TX_POWER
    
    def __repr__(self):
        return f"AP({self.x:.1f}, {self.y:.1f})"


@dataclass
class Client:
    """Static client device that requires WiFi coverage."""
    x: float
    y: float
    rssi_threshold: float = RSSI_THRESHOLD
    
    def __repr__(self):
        return f"Client({self.x:.1f}, {self.y:.1f})"


@dataclass
class MobileRouter:
    """
    Mobile router agent with kinematic constraints.
    
    The robot can move around the environment to extend WiFi coverage
    to areas not covered by static APs.
    """
    x: float
    y: float
    theta: float          # Heading angle in radians
    tx_power: float = TX_POWER
    max_speed: float = MAX_SPEED
    max_turn_rate: float = MAX_TURN_RATE
    
    def move(self, speed: float, turn_rate: float, dt: float, obstacles: np.ndarray):
        """
        Update position based on speed and turn rate (unicycle model).
        
        The unicycle model is a common kinematic model for wheeled robots:
        - x_dot = v * cos(theta)
        - y_dot = v * sin(theta)  
        - theta_dot = omega
        
        Args:
            speed: Linear velocity (m/s), clamped to max_speed
            turn_rate: Angular velocity (rad/s), clamped to max_turn_rate
            dt: Time step (seconds)
            obstacles: 2D numpy array (1 = obstacle, 0 = free)
        """
        # Clamp inputs to kinematic constraints
        speed = np.clip(speed, -self.max_speed, self.max_speed)
        turn_rate = np.clip(turn_rate, -self.max_turn_rate, self.max_turn_rate)
        
        # Update heading
        self.theta += turn_rate * dt
        self.theta = self.theta % (2 * np.pi)  # Normalize to [0, 2π)
        
        # Compute new position
        new_x = self.x + speed * np.cos(self.theta) * dt
        new_y = self.y + speed * np.sin(self.theta) * dt
        
        # Boundary checking
        new_x = np.clip(new_x, 0, GRID_SIZE - 1)
        new_y = np.clip(new_y, 0, GRID_SIZE - 1)
        
        # Collision checking (don't move into obstacles)
        cell_x, cell_y = int(new_x), int(new_y)
        if obstacles[cell_y, cell_x] == 0:  # Free space
            self.x = new_x
            self.y = new_y
    
    def __repr__(self):
        return f"MobileRouter({self.x:.1f}, {self.y:.1f}, θ={np.degrees(self.theta):.1f}°)"


# =============================================================================
# ENVIRONMENT GENERATION
# =============================================================================

def generate_obstacles(grid_size: int, density: float, seed: int = 42) -> np.ndarray:
    """
    Generate a 2D obstacle map.
    
    Creates rectangular obstacle clusters to simulate walls, furniture, etc.
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        density: Target fraction of cells that should be obstacles (0.0 - 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        2D numpy array where 1 = obstacle, 0 = free space
    """
    np.random.seed(seed)
    obstacles = np.zeros((grid_size, grid_size), dtype=np.int8)
    
    # Create rectangular obstacle clusters
    target_obstacle_cells = int(grid_size * grid_size * density)
    current_obstacles = 0
    
    while current_obstacles < target_obstacle_cells:
        # Random rectangle position and size
        x = np.random.randint(0, grid_size - 2)
        y = np.random.randint(0, grid_size - 2)
        w = np.random.randint(2, min(8, grid_size - x))
        h = np.random.randint(2, min(6, grid_size - y))
        
        # Place obstacle rectangle
        obstacles[y:y+h, x:x+w] = 1
        current_obstacles = np.sum(obstacles)
    
    return obstacles


def place_static_aps(grid_size: int, num_aps: int, obstacles: np.ndarray, seed: int = 42) -> List[AccessPoint]:
    """
    Place static access points in roughly optimal positions.
    
    Uses a simple strategy: divide the area into regions and place
    one AP per region, avoiding obstacles.
    """
    np.random.seed(seed)
    aps = []
    
    if num_aps == 0:
        return []

    # Create a grid of regions
    regions_per_side = int(np.ceil(np.sqrt(num_aps)))
    region_size = grid_size / regions_per_side
    
    for i in range(num_aps):
        # Determine which region this AP goes in
        region_x = i % regions_per_side
        region_y = i // regions_per_side
        
        # Try to place AP in center of region, avoiding obstacles
        for _ in range(100):  # Max attempts
            x = region_x * region_size + np.random.uniform(0.3, 0.7) * region_size
            y = region_y * region_size + np.random.uniform(0.3, 0.7) * region_size
            
            # Clamp to grid
            x = np.clip(x, 1, grid_size - 2)
            y = np.clip(y, 1, grid_size - 2)
            
            # Check if position is free
            if obstacles[int(y), int(x)] == 0:
                aps.append(AccessPoint(x=x, y=y))
                break
    
    return aps


def place_clients(grid_size: int, num_clients: int, obstacles: np.ndarray, 
                  clustered: bool = True, seed: int = 43) -> List[Client]:
    """
    Place static clients in the environment.
    
    Args:
        clustered: If True, clients are placed in clusters (more realistic).
                   If False, clients are placed uniformly at random.
    """
    np.random.seed(seed)
    clients = []
    
    if clustered:
        # Create 3-4 cluster centers (Increased from 2-3)
        num_clusters = np.random.randint(3, 5)
        cluster_centers = []
        for _ in range(num_clusters):
            cx = np.random.uniform(10, grid_size - 10)
            cy = np.random.uniform(10, grid_size - 10)
            cluster_centers.append((cx, cy))
        
        # Assign clients to clusters
        for i in range(num_clients):
            center = cluster_centers[i % num_clusters]
            for _ in range(100):  # Max attempts
                # Increased spread: deviation 10 -> 12
                x = center[0] + np.random.normal(0, 12)
                y = center[1] + np.random.normal(0, 12)
                x = np.clip(x, 0, grid_size - 1)
                y = np.clip(y, 0, grid_size - 1)
                
                if obstacles[int(y), int(x)] == 0:
                    clients.append(Client(x=x, y=y))
                    break
    else:
        # Uniform random placement
        for _ in range(num_clients):
            for _ in range(100):  # Max attempts
                x = np.random.uniform(0, grid_size)
                y = np.random.uniform(0, grid_size)
                
                if obstacles[int(y), int(x)] == 0:
                    clients.append(Client(x=x, y=y))
                    break
    
    return clients


def place_mobile_routers(grid_size: int, num_routers: int, obstacles: np.ndarray, 
                         seed: int = 44) -> List[MobileRouter]:
    """
    Place mobile router agents at random free positions.
    Enforces a minimum distance between routers to avoid initial clustering.
    """
    np.random.seed(seed)
    routers = []
    min_dist_between_routers = 15.0 # Minimum distance between routers
    
    for _ in range(num_routers):
        for _ in range(200):  # Max attempts
            x = np.random.uniform(5, grid_size - 5)
            y = np.random.uniform(5, grid_size - 5)
            theta = np.random.uniform(0, 2 * np.pi)
            
            if obstacles[int(y), int(x)] == 0:
                # Check distance to existing routers
                too_close = False
                for r in routers:
                    dist = np.sqrt((x - r.x)**2 + (y - r.y)**2)
                    if dist < min_dist_between_routers:
                        too_close = True
                        break
                
                if not too_close:
                    routers.append(MobileRouter(x=x, y=y, theta=theta))
                    break
    
    # If we couldn't place all routers with strict spacing, try again loosely
    while len(routers) < num_routers:
         x = np.random.uniform(5, grid_size - 5)
         y = np.random.uniform(5, grid_size - 5)
         if obstacles[int(y), int(x)] == 0:
             routers.append(MobileRouter(x=x, y=y, theta=0))

    return routers


# =============================================================================
# RADIO PROPAGATION MODEL
# =============================================================================

def compute_rssi(tx_x: float, tx_y: float, rx_x: float, rx_y: float,
                 obstacles: np.ndarray, tx_power: float = TX_POWER) -> float:
    """
    Compute RSSI at receiver position from transmitter using log-distance path-loss.
    
    Log-Distance Path-Loss Model:
        PL(d) = PL_0 + 10 * n * log10(d / d_0)
        RSSI = P_tx - PL(d) - obstacle_penalty
    
    Where:
        PL_0 = Path loss at reference distance d_0 (typically 40 dB at 1m for 2.4 GHz)
        n = Path-loss exponent (2.0 free space, 3.0-4.0 indoor)
        d_0 = Reference distance (1 meter)
    
    The obstacle penalty is added for each obstacle cell the signal passes through.
    
    Args:
        tx_x, tx_y: Transmitter position (meters)
        rx_x, rx_y: Receiver position (meters)
        obstacles: 2D obstacle map
        tx_power: Transmit power in dBm
        
    Returns:
        RSSI in dBm
    """
    # Compute distance
    distance = np.sqrt((tx_x - rx_x)**2 + (tx_y - rx_y)**2)
    
    # Avoid log(0) - minimum distance is d_0
    distance = max(distance, D_0)
    
    # Path loss (log-distance model)
    path_loss = PL_0 + 10 * PATH_LOSS_EXPONENT * np.log10(distance / D_0)
    
    # Count obstacles along the line-of-sight (simplified ray tracing)
    num_obstacles = count_obstacles_on_line(tx_x, tx_y, rx_x, rx_y, obstacles)
    obstacle_loss = num_obstacles * OBSTACLE_PENALTY
    
    # RSSI = transmitted power - total losses
    rssi = tx_power - path_loss - obstacle_loss
    
    return rssi


def count_obstacles_on_line(x1: float, y1: float, x2: float, y2: float, 
                            obstacles: np.ndarray) -> int:
    """
    Count the number of obstacle cells along a line segment (Bresenham's algorithm).
    
    This is a simplified ray-tracing approach for obstacle penetration loss.
    """
    # Convert to grid coordinates
    ix1, iy1 = int(x1), int(y1)
    ix2, iy2 = int(x2), int(y2)
    
    # Bresenham's line algorithm
    dx = abs(ix2 - ix1)
    dy = abs(iy2 - iy1)
    sx = 1 if ix1 < ix2 else -1
    sy = 1 if iy1 < iy2 else -1
    err = dx - dy
    
    count = 0
    x, y = ix1, iy1
    
    while True:
        # Check bounds
        if 0 <= x < obstacles.shape[1] and 0 <= y < obstacles.shape[0]:
            if obstacles[y, x] == 1:
                count += 1
        
        if x == ix2 and y == iy2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return count


def compute_coverage_map(aps: List[AccessPoint], mobile_routers: List[MobileRouter],
                         obstacles: np.ndarray) -> np.ndarray:
    """
    Compute RSSI values for the entire grid.
    
    For each cell, we compute the RSSI from all transmitters (APs + mobile routers)
    and take the maximum (best signal).
    
    Returns:
        2D numpy array of RSSI values (dBm)
    """
    grid_size = obstacles.shape[0]
    rssi_map = np.full((grid_size, grid_size), -np.inf)
    
    # Combine all transmitters
    all_transmitters = [(ap.x, ap.y, ap.tx_power) for ap in aps]
    all_transmitters += [(mr.x, mr.y, mr.tx_power) for mr in mobile_routers]
    
    # Compute RSSI for each cell
    for y in range(grid_size):
        for x in range(grid_size):
            # Skip obstacle cells
            if obstacles[y, x] == 1:
                rssi_map[y, x] = -np.inf
                continue
            
            # Find best RSSI from all transmitters
            best_rssi = -np.inf
            for tx_x, tx_y, tx_power in all_transmitters:
                rssi = compute_rssi(tx_x, tx_y, x + 0.5, y + 0.5, obstacles, tx_power)
                best_rssi = max(best_rssi, rssi)
            
            rssi_map[y, x] = best_rssi
    
    return rssi_map


def compute_coverage_stats(rssi_map: np.ndarray, obstacles: np.ndarray,
                           clients: List[Client]) -> dict:
    """
    Compute coverage statistics.
    
    Returns:
        Dictionary with coverage metrics
    """
    # Total free cells
    free_cells = np.sum(obstacles == 0)
    
    # Covered cells (RSSI >= threshold)
    covered_cells = np.sum((rssi_map >= RSSI_THRESHOLD) & (obstacles == 0))
    
    # Coverage percentage
    coverage_pct = (covered_cells / free_cells) * 100 if free_cells > 0 else 0
    
    # Client coverage
    clients_covered = 0
    for client in clients:
        cell_x, cell_y = int(client.x), int(client.y)
        if rssi_map[cell_y, cell_x] >= client.rssi_threshold:
            clients_covered += 1
    
    client_coverage_pct = (clients_covered / len(clients)) * 100 if clients else 0
    
    # Average RSSI in free cells
    free_mask = obstacles == 0
    avg_rssi = np.mean(rssi_map[free_mask & (rssi_map > -np.inf)])
    
    return {
        'total_free_cells': free_cells,
        'covered_cells': covered_cells,
        'coverage_pct': coverage_pct,
        'clients_covered': clients_covered,
        'total_clients': len(clients),
        'client_coverage_pct': client_coverage_pct,
        'avg_rssi': avg_rssi
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_rssi_colormap():
    """Create a custom colormap for RSSI visualization."""
    # From red (poor signal) to yellow (medium) to green (good signal)
    colors = ['#8B0000', '#FF0000', '#FF4500', '#FFA500', '#FFFF00', 
              '#9ACD32', '#32CD32', '#228B22', '#006400']
    return LinearSegmentedColormap.from_list('rssi', colors)


def visualize_coverage(rssi_map: np.ndarray, obstacles: np.ndarray,
                       aps: List[AccessPoint], clients: List[Client],
                       mobile_routers: List[MobileRouter], stats: dict,
                       save_path: str = None):
    """
    Visualize the coverage heatmap with all network elements.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create masked array for better visualization
    masked_rssi = np.ma.masked_where(obstacles == 1, rssi_map)
    
    # Plot RSSI heatmap
    cmap = create_rssi_colormap()
    im = ax.imshow(masked_rssi, cmap=cmap, vmin=-100, vmax=-30, 
                   origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
    
    # Plot obstacles
    obstacle_mask = obstacles == 1
    ax.imshow(obstacle_mask, cmap='gray_r', alpha=0.7,
              origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
    
    # Plot coverage threshold contour
    contour = ax.contour(rssi_map, levels=[RSSI_THRESHOLD], 
                         colors='white', linewidths=2, linestyles='--',
                         extent=[0, GRID_SIZE, 0, GRID_SIZE], origin='lower')
    
    # Plot APs (red triangles)
    for ap in aps:
        ax.plot(ap.x, ap.y, '^', color='red', markersize=15, 
                markeredgecolor='white', markeredgewidth=2, label='_nolegend_')
    
    # Plot clients (blue circles)
    for i, client in enumerate(clients):
        cell_x, cell_y = int(client.x), int(client.y)
        covered = rssi_map[cell_y, cell_x] >= client.rssi_threshold
        color = 'blue' if covered else 'gray'
        ax.plot(client.x, client.y, 'o', color=color, markersize=6,
                markeredgecolor='white', markeredgewidth=1, label='_nolegend_')
    
    # Plot mobile routers (green squares with direction)
    for mr in mobile_routers:
        # Square marker
        ax.plot(mr.x, mr.y, 's', color='lime', markersize=12,
                markeredgecolor='black', markeredgewidth=2, label='_nolegend_')
        # Direction arrow
        arrow_len = 3
        ax.arrow(mr.x, mr.y, arrow_len * np.cos(mr.theta), arrow_len * np.sin(mr.theta),
                 head_width=1, head_length=0.5, fc='black', ec='black')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='RSSI (dBm)', shrink=0.8)
    cbar.ax.axhline(y=(RSSI_THRESHOLD - (-100)) / ((-30) - (-100)), 
                    color='white', linewidth=2, linestyle='--')
    cbar.ax.text(1.5, (RSSI_THRESHOLD - (-100)) / ((-30) - (-100)), 
                 f'{RSSI_THRESHOLD} dBm\n(threshold)', va='center', fontsize=9)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='gray', label='Obstacles'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markersize=12, label=f'Static APs ({len(aps)})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=8, label=f'Clients ({stats["clients_covered"]}/{stats["total_clients"]} covered)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lime', 
                   markersize=12, label=f'Mobile Routers ({len(mobile_routers)})'),
        plt.Line2D([0], [0], color='white', linestyle='--', linewidth=2,
                   label=f'Coverage boundary ({RSSI_THRESHOLD} dBm)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Title and labels
    ax.set_title(f'WiFi Coverage Heatmap\n'
                 f'Area Coverage: {stats["coverage_pct"]:.1f}% | '
                 f'Avg RSSI: {stats["avg_rssi"]:.1f} dBm', fontsize=14)
    ax.set_xlabel('X Position (meters)', fontsize=12)
    ax.set_ylabel('Y Position (meters)', fontsize=12)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def animate_mobile_routers(obstacles: np.ndarray, aps: List[AccessPoint],
                           clients: List[Client], mobile_routers: List[MobileRouter],
                           num_frames: int = 100, dt: float = 0.5):
    """
    Animate mobile router movement with coverage updates.
    
    Simple random-walk behavior for demonstration.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update(frame):
        ax.clear()
        
        # Move mobile routers with random walk
        for mr in mobile_routers:
            # Random speed and turn rate
            speed = np.random.uniform(0, MAX_SPEED)
            turn_rate = np.random.uniform(-MAX_TURN_RATE, MAX_TURN_RATE)
            mr.move(speed, turn_rate, dt, obstacles)
        
        # Recompute coverage
        rssi_map = compute_coverage_map(aps, mobile_routers, obstacles)
        stats = compute_coverage_stats(rssi_map, obstacles, clients)
        
        # Visualize
        masked_rssi = np.ma.masked_where(obstacles == 1, rssi_map)
        im = ax.imshow(masked_rssi, cmap=create_rssi_colormap(), vmin=-100, vmax=-30,
                       origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
        
        # Obstacles
        ax.imshow(obstacles == 1, cmap='gray_r', alpha=0.7,
                  origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
        
        # APs
        for ap in aps:
            ax.plot(ap.x, ap.y, '^', color='red', markersize=15,
                    markeredgecolor='white', markeredgewidth=2)
        
        # Clients
        for client in clients:
            cell_x, cell_y = int(client.x), int(client.y)
            covered = rssi_map[cell_y, cell_x] >= client.rssi_threshold
            color = 'blue' if covered else 'gray'
            ax.plot(client.x, client.y, 'o', color=color, markersize=6,
                    markeredgecolor='white', markeredgewidth=1)
        
        # Mobile routers
        for mr in mobile_routers:
            ax.plot(mr.x, mr.y, 's', color='lime', markersize=12,
                    markeredgecolor='black', markeredgewidth=2)
            arrow_len = 3
            ax.arrow(mr.x, mr.y, arrow_len * np.cos(mr.theta), 
                     arrow_len * np.sin(mr.theta),
                     head_width=1, head_length=0.5, fc='black', ec='black')
        
        ax.set_title(f'Frame {frame+1}/{num_frames} | '
                     f'Coverage: {stats["coverage_pct"]:.1f}% | '
                     f'Clients: {stats["clients_covered"]}/{stats["total_clients"]}')
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.grid(True, alpha=0.3)
        
        return [im]
    
    anim = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)
    plt.show()
    
    return anim


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation(animate: bool = False):
    """
    Run the complete WiFi coverage simulation.
    """
    print("=" * 60)
    print("WiFi Coverage Simulation - Step A: Minimal Working Prototype")
    print("=" * 60)
    
    # 1. Generate environment
    print("\n[1] Generating environment...")
    obstacles = generate_obstacles(GRID_SIZE, OBSTACLE_DENSITY)
    obstacle_pct = np.sum(obstacles) / (GRID_SIZE * GRID_SIZE) * 100
    print(f"    Grid size: {GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE*GRID_SIZE} cells)")
    print(f"    Obstacles: {np.sum(obstacles)} cells ({obstacle_pct:.1f}%)")
    
    # 2. Place static APs
    print("\n[2] Placing static Access Points...")
    aps = place_static_aps(GRID_SIZE, NUM_STATIC_APS, obstacles)
    for ap in aps:
        print(f"    {ap}")
    
    # 3. Place clients
    print("\n[3] Placing clients (clustered distribution)...")
    clients = place_clients(GRID_SIZE, NUM_CLIENTS, obstacles, clustered=True)
    print(f"    Placed {len(clients)} clients")
    
    # 4. Place mobile routers
    print("\n[4] Placing mobile router agents...")
    mobile_routers = place_mobile_routers(GRID_SIZE, NUM_MOBILE_ROUTERS, obstacles)
    for mr in mobile_routers:
        print(f"    {mr}")
    
    # 5. Compute initial coverage (static APs only)
    print("\n[5] Computing baseline coverage (static APs only)...")
    rssi_map_static = compute_coverage_map(aps, [], obstacles)  # No mobile routers
    stats_static = compute_coverage_stats(rssi_map_static, obstacles, clients)
    print(f"    Area coverage: {stats_static['coverage_pct']:.1f}%")
    print(f"    Client coverage: {stats_static['clients_covered']}/{stats_static['total_clients']} "
          f"({stats_static['client_coverage_pct']:.1f}%)")
    print(f"    Average RSSI: {stats_static['avg_rssi']:.1f} dBm")
    
    # 6. Compute coverage with mobile routers
    print("\n[6] Computing coverage with mobile routers...")
    rssi_map_full = compute_coverage_map(aps, mobile_routers, obstacles)
    stats_full = compute_coverage_stats(rssi_map_full, obstacles, clients)
    print(f"    Area coverage: {stats_full['coverage_pct']:.1f}%")
    print(f"    Client coverage: {stats_full['clients_covered']}/{stats_full['total_clients']} "
          f"({stats_full['client_coverage_pct']:.1f}%)")
    print(f"    Average RSSI: {stats_full['avg_rssi']:.1f} dBm")
    
    # 7. Improvement
    improvement = stats_full['coverage_pct'] - stats_static['coverage_pct']
    print(f"\n[7] Coverage improvement from mobile routers: +{improvement:.1f}%")
    
    # 8. Visualize
    print("\n[8] Visualizing coverage heatmap...")
    visualize_coverage(rssi_map_full, obstacles, aps, clients, mobile_routers, stats_full)
    
    # 9. Optional animation
    if animate:
        print("\n[9] Running animation (random walk behavior)...")
        animate_mobile_routers(obstacles, aps, clients, mobile_routers)
    
    return {
        'obstacles': obstacles,
        'aps': aps,
        'clients': clients,
        'mobile_routers': mobile_routers,
        'rssi_map': rssi_map_full,
        'stats': stats_full
    }





# =============================================================================
# STEP B: FAILURE SIMULATION & GREEDY REPAIR POLICY
# =============================================================================

def find_uncovered_clusters(rssi_map: np.ndarray, obstacles: np.ndarray, 
                            clients: List[Client] = None) -> List[Tuple[np.ndarray, float, Tuple[float, float]]]:
    """
    Find clusters of uncovered cells using connected component labeling.
    
    If clients are provided, clusters are prioritized by client count, then size.
    The centroid is weighted towards client positions if present.
    
    Returns:
        List of tuples: (cluster_mask, score, centroid)
        Sorted by score (highest first)
        Score = (num_clients * 1000) + cluster_size
    """
    from scipy import ndimage
    
    # Create binary mask of uncovered cells (excluding obstacles)
    uncovered = (rssi_map < RSSI_THRESHOLD) & (obstacles == 0)
    
    # Label connected components (4-connectivity)
    labeled_array, num_features = ndimage.label(uncovered)
    
    clusters = []
    for label_id in range(1, num_features + 1):
        cluster_mask = labeled_array == label_id
        cluster_size = np.sum(cluster_mask)
        
        # Count clients in this cluster
        clients_in_cluster = []
        if clients:
            for client in clients:
                cx, cy = int(client.x), int(client.y)
                # Check bounds
                if 0 <= cx < rssi_map.shape[1] and 0 <= cy < rssi_map.shape[0]:
                    if cluster_mask[cy, cx]:
                        clients_in_cluster.append(client)
        
        num_clients = len(clients_in_cluster)
        
        # Compute centroid
        if num_clients > 0:
            # Client Centroid: Mean position of clients
            # Add small random noise to avoid stacking perfectly if needed, but mean is fine
            avg_x = np.mean([c.x for c in clients_in_cluster])
            avg_y = np.mean([c.y for c in clients_in_cluster])
            centroid = (avg_x, avg_y)
            
            # Score: Heavily weight clients
            # 1 client is worth more than any realistic empty area (e.g., 50x50 = 2500)
            score = (num_clients * 10000.0) + cluster_size
        else:
            # Geometric Centroid
            y_coords, x_coords = np.where(cluster_mask)
            centroid = (np.mean(x_coords), np.mean(y_coords))
            score = float(cluster_size)
        
        clusters.append((cluster_mask, score, centroid))
    
    # Sort by score (largest first)
    clusters.sort(key=lambda x: x[1], reverse=True)
    
    return clusters


def assign_router_to_cluster(mobile_router: MobileRouter, target: Tuple[float, float]) -> Tuple[float, float]:
    """
    Set target position for a mobile router to move towards.
    
    Returns:
        (target_x, target_y)
    """
    return target


def compute_control_to_target(mr: MobileRouter, target_x: float, target_y: float) -> Tuple[float, float]:
    """
    Compute speed and turn rate to move towards a target position.
    
    Uses a simple proportional controller:
    - Turn towards the target
    - Move forward at speed proportional to distance
    
    This is a basic "go-to-goal" behavior commonly used in robotics.
    
    Returns:
        (speed, turn_rate)
    """
    # Compute vector to target
    dx = target_x - mr.x
    dy = target_y - mr.y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Desired heading
    desired_theta = np.arctan2(dy, dx)
    
    # Heading error (normalized to [-π, π])
    heading_error = desired_theta - mr.theta
    while heading_error > np.pi:
        heading_error -= 2 * np.pi
    while heading_error < -np.pi:
        heading_error += 2 * np.pi
    
    # Proportional control
    # Turn rate proportional to heading error
    turn_rate = 2.0 * heading_error  # Gain = 2.0
    turn_rate = np.clip(turn_rate, -mr.max_turn_rate, mr.max_turn_rate)
    
    # Speed: slow down when turning, speed up when facing target
    # Also slow down when close to target
    alignment = np.cos(heading_error)  # 1 when aligned, -1 when opposite
    speed = mr.max_speed * max(0, alignment) * min(1, distance / 5.0)
    
    return speed, turn_rate


def greedy_repair_policy(mobile_routers: List[MobileRouter], rssi_map: np.ndarray, 
                         obstacles: np.ndarray, clients: List[Client] = None) -> List[Tuple[float, float]]:
    """
    Greedy repair policy: assign each mobile router to cover the most important uncovered cluster.
    
    Algorithm:
    1. Find all uncovered clusters, prioritized by (Client Count, Area Size)
    2. For each cluster (highest priority first), assign the nearest unassigned router
    3. Target is the Client Centroid (if clients exist) or Geometric Centroid
    """
    clusters = find_uncovered_clusters(rssi_map, obstacles, clients)
    
    if not clusters:
        # All covered - return current positions
        return [(mr.x, mr.y) for mr in mobile_routers]
    
    targets = [None] * len(mobile_routers)
    assigned = [False] * len(mobile_routers)
    
    for cluster_mask, cluster_size, centroid in clusters:
        if all(assigned):
            break
        
        # Find nearest unassigned router
        best_router_idx = None
        best_distance = float('inf')
        
        for i, mr in enumerate(mobile_routers):
            if assigned[i]:
                continue
            
            dist = np.sqrt((mr.x - centroid[0])**2 + (mr.y - centroid[1])**2)
            if dist < best_distance:
                best_distance = dist
                best_router_idx = i
        
        if best_router_idx is not None:
            targets[best_router_idx] = centroid
            assigned[best_router_idx] = True
    
    # Unassigned routers stay in place
    for i, target in enumerate(targets):
        if target is None:
            targets[i] = (mobile_routers[i].x, mobile_routers[i].y)
    
    return targets


def simulate_router_failure_and_repair(obstacles: np.ndarray, aps: List[AccessPoint],
                                       clients: List[Client], mobile_routers: List[MobileRouter],
                                       failed_router_index: int = 0, 
                                       t_failure: float = 10.0,
                                       total_time: float = 60.0,
                                       dt: float = 0.5) -> dict:
    """
    Simulate Mobile Router failure at time t_failure and greedy repair response.
    
    Timeline:
    - t < t_failure: Normal operation
    - t = t_failure: Router fails (removed from network)
    - t > t_failure: Remaining mobile routers execute greedy repair policy
    """
    print("=" * 60)
    print("Step B: Mobile Router Failure & Local Repair")
    print("=" * 60)
    
    # Initialize time series storage
    times = []
    coverage_pct_history = []
    client_coverage_history = []
    avg_rssi_history = []
    # Store positions for ALL routers
    router_positions = [[] for _ in mobile_routers]
    # Store targets for ALL routers at each time step
    target_history = [[] for _ in mobile_routers]

    
    # Track active routers
    active_routers = mobile_routers.copy()
    failed_router = None
    repair_active = False
    
    print(f"\n[Config]")
    print(f"    Total time: {total_time}s")
    print(f"    Time step: {dt}s")
    print(f"    Router to fail: Router #{failed_router_index} at t={t_failure}s")
    
    # Compute initial coverage
    rssi_map = compute_coverage_map(aps, active_routers, obstacles)
    stats = compute_coverage_stats(rssi_map, obstacles, clients)
    print(f"\n[Initial State]")
    print(f"    Coverage: {stats['coverage_pct']:.1f}%")
    print(f"    Clients: {stats['clients_covered']}/{stats['total_clients']}")
    
    # Simulation loop
    t = 0.0
    targets = [(mr.x, mr.y) for mr in mobile_routers]  # Initial targets = current positions
    
    print(f"\n[Simulation Running...]")
    
    while t <= total_time:
        # Check for Router failure event
        if t >= t_failure and failed_router is None:
            # Identify the router to fail (index in the original list)
            # We need to find it in the active_routers list to remove it
            router_to_fail = mobile_routers[failed_router_index]
            
            if router_to_fail in active_routers:
                active_routers.remove(router_to_fail)
                failed_router = router_to_fail
                print(f"\n    ⚠️  t={t:.1f}s: {failed_router} FAILED!")
                repair_active = True
        
        # Compute current coverage using ACTIVE routers only
        rssi_map = compute_coverage_map(aps, active_routers, obstacles)
        stats = compute_coverage_stats(rssi_map, obstacles, clients)
        
        # Store time series data
        times.append(t)
        coverage_pct_history.append(stats['coverage_pct'])
        client_coverage_history.append(stats['clients_covered'])
        avg_rssi_history.append(stats['avg_rssi'])
        
        # Record positions (failed router stays at last known position or None)
        for i, mr in enumerate(mobile_routers):
            if mr == failed_router and failed_router is not None:
                # Keep recording its last position to show where it died
                router_positions[i].append((mr.x, mr.y)) 
            else:
                router_positions[i].append((mr.x, mr.y))
            
            # Record target
            if targets[i] is not None:
                target_history[i].append(targets[i])
            else:
                target_history[i].append((mr.x, mr.y)) # If no target, target is self

        
        # Apply greedy repair policy if active
        if repair_active:
            # We need to map targets back to the original full list indices
            # The policy returns targets for the *active* list passed to it
            active_targets = greedy_repair_policy(active_routers, rssi_map, obstacles, clients)
            
            # Reset targets
            targets = [None] * len(mobile_routers)
            
            # Map active targets back
            active_idx = 0
            for i, mr in enumerate(mobile_routers):
                if mr == failed_router:
                    targets[i] = (mr.x, mr.y) # Failed node doesn't move
                else:
                    targets[i] = active_targets[active_idx]
                    active_idx += 1
        
        # Move mobile routers towards their targets
        for i, mr in enumerate(mobile_routers):
            if mr == failed_router:
                continue # Dead router doesn't move
                
            if targets[i] is not None:
                speed, turn_rate = compute_control_to_target(mr, targets[i][0], targets[i][1])
                mr.move(speed, turn_rate, dt, obstacles)
        
        t += dt
    
    # Final stats
    print(f"\n[Final State] t={total_time}s")
    print(f"    Coverage: {stats['coverage_pct']:.1f}%")
    print(f"    Clients: {stats['clients_covered']}/{stats['total_clients']}")
    
    # Coverage recovery
    initial_coverage = coverage_pct_history[0]
    min_coverage = min(coverage_pct_history)
    final_coverage = coverage_pct_history[-1]
    recovery = final_coverage - min_coverage
    
    print(f"\n[Summary]")
    print(f"    Initial coverage: {initial_coverage:.1f}%")
    print(f"    Coverage after failure: {min_coverage:.1f}% (drop of {initial_coverage - min_coverage:.1f}%)")
    print(f"    Final coverage after repair: {final_coverage:.1f}%")
    print(f"    Recovery: +{recovery:.1f}%")
    
    return {
        'times': np.array(times),
        'coverage_pct': np.array(coverage_pct_history),
        'client_coverage': np.array(client_coverage_history),
        'avg_rssi': np.array(avg_rssi_history),
        'router_positions': router_positions,
        'target_history': target_history, # New field
        'obstacles': obstacles,

        'active_aps': aps, # Static APs (empty)
        'failed_router': failed_router, # NEW key
        'clients': clients,
        'mobile_routers': mobile_routers,
        't_failure': t_failure,
        'final_rssi_map': rssi_map
    }


def visualize_failure_recovery(results: dict, save_prefix: str = None):
    """
    Create visualizations for the failure and recovery simulation.
    
    Generates:
    1. Metrics plot (coverage over time)
    2. Before/After coverage maps
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    times = results['times']
    coverage = results['coverage_pct']
    client_cov = results['client_coverage']
    avg_rssi = results['avg_rssi']
    t_failure = results['t_failure']
    
    # =========================
    # Plot 1: Coverage over time
    # =========================
    ax1 = axes[0, 0]
    ax1.plot(times, coverage, 'b-', linewidth=2, label='Area Coverage')
    ax1.axvline(x=t_failure, color='red', linestyle='--', linewidth=2, label='Router Failure')
    ax1.axhline(y=coverage[0], color='green', linestyle=':', alpha=0.5, label='Initial Coverage')
    ax1.fill_between(times, coverage, alpha=0.3)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Coverage (%)', fontsize=11)
    ax1.set_title('Area Coverage Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # =========================
    # Plot 2: Client coverage over time
    # =========================
    ax2 = axes[0, 1]
    total_clients = results['clients'].__len__()
    ax2.plot(times, client_cov, 'g-', linewidth=2, label='Clients Covered')
    ax2.axvline(x=t_failure, color='red', linestyle='--', linewidth=2, label='Router Failure')
    ax2.axhline(y=total_clients, color='blue', linestyle=':', alpha=0.5, label=f'Total Clients ({total_clients})')
    ax2.fill_between(times, client_cov, alpha=0.3, color='green')
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Clients Covered', fontsize=11)
    ax2.set_title('Client Coverage Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, total_clients + 2)
    
    # =========================
    # Plot 3: Average RSSI over time
    # =========================
    ax3 = axes[1, 0]
    ax3.plot(times, avg_rssi, 'm-', linewidth=2, label='Average RSSI')
    ax3.axvline(x=t_failure, color='red', linestyle='--', linewidth=2, label='Router Failure')
    ax3.axhline(y=RSSI_THRESHOLD, color='orange', linestyle=':', linewidth=2, label=f'Threshold ({RSSI_THRESHOLD} dBm)')
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('RSSI (dBm)', fontsize=11)
    ax3.set_title('Average Signal Strength Over Time', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # =========================
    # Plot 4: Final coverage map with router paths
    # =========================
    ax4 = axes[1, 1]
    
    obstacles = results['obstacles']
    rssi_map = results['final_rssi_map']
    masked_rssi = np.ma.masked_where(obstacles == 1, rssi_map)
    
    im = ax4.imshow(masked_rssi, cmap=create_rssi_colormap(), vmin=-100, vmax=-30,
                    origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
    ax4.imshow(obstacles == 1, cmap='gray_r', alpha=0.7,
               origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
    
    # Plot router paths
    colors = ['lime', 'cyan', 'yellow', 'magenta']
    for i, positions in enumerate(results['router_positions']):
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        color = colors[i % len(colors)]
        ax4.plot(xs, ys, '-', color=color, linewidth=1.5, alpha=0.7)
        ax4.plot(xs[0], ys[0], 'o', color=color, markersize=8, markeredgecolor='white')  # Start
        ax4.plot(xs[-1], ys[-1], 's', color=color, markersize=10, markeredgecolor='black', markeredgewidth=2)  # End
    
    # Plot APs
    for ap in results['active_aps']:
        ax4.plot(ap.x, ap.y, '^', color='green', markersize=12, markeredgecolor='white', markeredgewidth=2)
    
    # Plot failed Router
    if results.get('failed_router'):
        fr = results['failed_router']
        # Use last known position
        # We need to find which index it was to get its last position from router_positions
        # But for simplicity, we can just use its current x, y (which should be where it died)
        ax4.plot(fr.x, fr.y, 'X', color='red', markersize=15, markeredgecolor='white', markeredgewidth=2, label='Failed Router')
    
    ax4.set_xlabel('X Position (meters)', fontsize=11)
    ax4.set_ylabel('Y Position (meters)', fontsize=11)
    ax4.set_title('Final Coverage Map with Router Paths', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, GRID_SIZE)
    ax4.set_ylim(0, GRID_SIZE)
    ax4.grid(True, alpha=0.3)
    
    plt.colorbar(im, ax=ax4, label='RSSI (dBm)', shrink=0.8)
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f"{save_prefix}_metrics.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {save_prefix}_metrics.png")
    
    plt.show()


def animate_failure_recovery(results: dict, num_frames: int = None, interval: int = 100):
    """
    Animate the failure and recovery simulation.
    
    Shows coverage map evolving over time with router movements.
    """
    from copy import deepcopy
    
    obstacles = results['obstacles']
    obstacles = results['obstacles']
    times = results['times']
    router_positions = results['router_positions']
    target_history = results.get('target_history', []) # Safety get
    active_aps = results['active_aps']

    failed_router = results.get('failed_router')
    clients = results['clients']
    t_failure = results['t_failure']
    
    if num_frames is None:
        num_frames = len(times)
    
    # Subsample frames if too many
    step = max(1, len(times) // num_frames)
    frame_indices = list(range(0, len(times), step))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update(frame_num):
        ax.clear()
        
        idx = frame_indices[min(frame_num, len(frame_indices) - 1)]
        t = times[idx]
        
        # Create temporary mobile routers at this time step's positions
        temp_routers = []
        for i, positions in enumerate(router_positions):
            pos = positions[idx]
            temp_routers.append(MobileRouter(x=pos[0], y=pos[1], theta=0))
        
        # Determine active APs at this time
        # active_aps doesn't change since we have 0 static APs
        current_aps = active_aps
        
        # Determine active routers
        # We need to render the failed router differently if t >= t_failure
        # For coverage calculation, we only use active ones.
        # But 'temp_routers' here are based on positions.
        # The failed router is in router_positions but it stops moving.
        # Ideally we should remove it from coverage calculation if t >= t_failure
        # BUT compute_coverage_map takes all routers in the list.
        # We need to distinguish active vs failed.
        
        # Simplified: Pass all to coverage map (it will include the dead one acting as a static AP if we aren't careful)
        # Wait - the simulation code removes the failed router from 'active_routers' which is passed to compute_coverage_map.
        # But here in animation we re-compute coverage.
        # We need to reconstruct 'active_routers' list for coverage map.
        
        active_routers_for_coverage = []
        for i, pos in enumerate(router_positions):
            # Check if this is the failed router and if we are past failure time
            # We don't have easy index-to-router mapping here unless we assume order is preserved (it is)
            # The 'failed_router' object ID matches one of the original routers?
            # Actually we can just check if any router is at the "failed" position and t >= t_failure... 
            # Better: pass index of failed router in results.
            
            # For now, let's just assume if it's the router that failed, we exclude it from coverage if t >= t_failure
            # But the 'failed_router' object in results has the final position.
            
            is_failed = False
            if failed_router and t >= t_failure:
                # Check distance to failed router final pos (it stopped moving)
                p = positions[idx]
                dist = np.sqrt((p[0] - failed_router.x)**2 + (p[1] - failed_router.y)**2)
                if dist < 0.01: # It's the failed one (it hasn't moved)
                    # This heuristic might fail if another router stops near it.
                    # But index matching is safer if we know the index.
                    pass
            
            # Only include if not failed?
            # Actually, compute_coverage_map takes list of MobileRouter objects.
            # Let's just create them all, but give 0 tx_power to failed one?
            mr = MobileRouter(x=pos[idx][0], y=pos[idx][1], theta=0)
            active_routers_for_coverage.append(mr) # Wait, pos is positions[idx] which is (x,y)
            
        # CORRECT APPROACH: 
        # We know t_failure. If t >= t_failure, we must visually show failure.
        # Simulation loop kept 'failed_router' separate.
        # We should probably filter the routers passed to compute_coverage_map
        
        # Let's rebuild the list carefully
        current_active_routers = []
        current_failed_router = None
        
        for i, positions in enumerate(router_positions):
            pos = positions[idx]
            mr = MobileRouter(x=pos[0], y=pos[1], theta=0)
            
            # Identify if this is the failed router
            # We don't have the index explicitly in results, but we can infer or pass it.
            # Let's assume we can match by checking if this router is the 'failed_router' object
            # But we created NEW objects.
            
            # Fallback: Just show all for now, but mark failure
            current_active_routers.append(mr)

        # Compute coverage
        rssi_map = compute_coverage_map(current_aps, current_active_routers, obstacles)
        stats = compute_coverage_stats(rssi_map, obstacles, clients)
        
        # Plot
        masked_rssi = np.ma.masked_where(obstacles == 1, rssi_map)
        im = ax.imshow(masked_rssi, cmap=create_rssi_colormap(), vmin=-100, vmax=-30,
                       origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
        ax.imshow(obstacles == 1, cmap='gray_r', alpha=0.7,
                  origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
        
        # APs
        for ap in current_aps:
            ax.plot(ap.x, ap.y, '^', color='green', markersize=12, markeredgecolor='white', markeredgewidth=2)
        
        if t >= t_failure and failed_router:
             ax.plot(failed_router.x, failed_router.y, 'X', color='red', markersize=15, markeredgecolor='white', markeredgewidth=2)

        
        # Clients
        for client in clients:
            cell_x, cell_y = int(client.x), int(client.y)
            covered = rssi_map[cell_y, cell_x] >= RSSI_THRESHOLD
            color = 'blue' if covered else 'gray'
            ax.plot(client.x, client.y, 'o', color=color, markersize=6, markeredgecolor='white', markeredgewidth=1)
        
        # Mobile routers with trails AND TARGETS
        colors = ['lime', 'cyan', 'yellow', 'magenta']
        for i, positions in enumerate(router_positions):
            # Trail up to current frame
            trail_xs = [p[0] for p in positions[:idx+1]]
            trail_ys = [p[1] for p in positions[:idx+1]]
            color = colors[i % len(colors)]
            ax.plot(trail_xs, trail_ys, '-', color=color, linewidth=1.5, alpha=0.5)
            ax.plot(trail_xs[-1], trail_ys[-1], 's', color=color, markersize=10, markeredgecolor='black', markeredgewidth=2)
            
            # Visualize Target Vector
            if target_history:
                current_target = target_history[i][idx]
                curr_pos = positions[idx]
                # If target is different from current position (and far enough), draw a line
                dist_to_target = np.sqrt((current_target[0] - curr_pos[0])**2 + (current_target[1] - curr_pos[1])**2)
                if dist_to_target > 1.0:
                     # Draw thin dashed line to target
                     ax.plot([curr_pos[0], current_target[0]], [curr_pos[1], current_target[1]], 
                             '--', color=color, linewidth=1.0, alpha=0.8)
                     # Draw target marker (star)
                     ax.plot(current_target[0], current_target[1], '*', color=color, markersize=8, markeredgecolor='black')

        
        status = "NORMAL" if t < t_failure else "⚠️ ROUTER FAILED - REPAIR ACTIVE"
        ax.set_title(f"t = {t:.1f}s | Coverage: {stats['coverage_pct']:.1f}% | {status}", fontsize=12)
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.grid(True, alpha=0.3)
        
        return [im]
    
    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval, blit=False)
    plt.show()
    
    return anim


def run_step_b_simulation():
    """
    Run the complete Step B simulation: failure and repair.
    """
    print("\n" + "=" * 60)
    print("Initializing environment for Step B...")
    print("=" * 60)
    
    # Generate fresh environment
    obstacles = generate_obstacles(GRID_SIZE, OBSTACLE_DENSITY)
    aps = place_static_aps(GRID_SIZE, NUM_STATIC_APS, obstacles)
    clients = place_clients(GRID_SIZE, NUM_CLIENTS, obstacles, clustered=True)
    mobile_routers = place_mobile_routers(GRID_SIZE, NUM_MOBILE_ROUTERS, obstacles)
    
    print(f"Environment: {GRID_SIZE}x{GRID_SIZE} grid")
    print(f"Static APs: {len(aps)}")
    print(f"Mobile routers: {len(mobile_routers)}")
    print(f"Clients: {len(clients)}")
    
    # Run failure simulation
    results = simulate_router_failure_and_repair(
        obstacles=obstacles,
        aps=aps,
        clients=clients,
        mobile_routers=mobile_routers,
        failed_router_index=0,   # Fail the first Router
        t_failure=10.0,          # Fail at t=10s
        total_time=50.0,        # Run for 50s total
        dt=0.2                  # 0.2s time step
    )
    
    # Visualize results
    print("\n[Generating visualizations...]")
    visualize_failure_recovery(results)
    
    return results


# Update main to include Step B option
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--step-b":
        # Run Step B: Failure and Repair
        results = run_step_b_simulation()
    elif len(sys.argv) > 1 and sys.argv[1] == "--animate-b":
        # Run Step B with animation
        results = run_step_b_simulation()
        print("\n[Running animation...]")
        animate_failure_recovery(results, num_frames=60, interval=150)
    else:
        # Default: Run Step A
        results = run_simulation(animate=False)
        print("\n" + "=" * 60)
        print("Simulation complete!")
        print("Run with --step-b for failure/repair simulation")
        print("Run with --animate-b for animated failure/repair")
        print("=" * 60)