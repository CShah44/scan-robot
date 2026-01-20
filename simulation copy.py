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
NUM_STATIC_APS = 3       # Number of static access points
NUM_CLIENTS = 20         # Number of static clients
NUM_MOBILE_ROUTERS = 4   # Number of mobile router agents

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
        # Create 2-3 cluster centers
        num_clusters = np.random.randint(2, 4)
        cluster_centers = []
        for _ in range(num_clusters):
            cx = np.random.uniform(10, grid_size - 10)
            cy = np.random.uniform(10, grid_size - 10)
            cluster_centers.append((cx, cy))
        
        # Assign clients to clusters
        for i in range(num_clients):
            center = cluster_centers[i % num_clusters]
            for _ in range(100):  # Max attempts
                x = center[0] + np.random.normal(0, 5)
                y = center[1] + np.random.normal(0, 5)
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
    """
    np.random.seed(seed)
    routers = []
    
    for _ in range(num_routers):
        for _ in range(100):  # Max attempts
            x = np.random.uniform(5, grid_size - 5)
            y = np.random.uniform(5, grid_size - 5)
            theta = np.random.uniform(0, 2 * np.pi)
            
            if obstacles[int(y), int(x)] == 0:
                routers.append(MobileRouter(x=x, y=y, theta=theta))
                break
    
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


if __name__ == "__main__":
    # Run the simulation
    results = run_simulation(animate=False)
    
    print("\n" + "=" * 60)
    print("Simulation complete! You can modify parameters at the top of the file.")
    print("=" * 60)
