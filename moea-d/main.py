
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .moead import MOEADSolver
from .config import *
import os

def main():
    print("Starting MOEA/D Optimization...")
    solver = MOEADSolver()
    final_pop = solver.solve()
    print("Optimization Completed.")
    
    # 1. Pareto Front Plot (3D)
    # Filter for non-dominated solutions roughly (or just plot all)
    # MOEA/D gives a set of solutions approximating the front.
    # We can plot all of them.
    
    all_objs = np.array([ind.objectives for ind in final_pop])
    
    # Objectives: [-Coverage, -Quality, Overlap]
    # For plotting we want: Coverage, Quality, Overlap
    cov = -all_objs[:, 0]
    qual = -all_objs[:, 1]
    over = all_objs[:, 2]
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(cov, qual, over, c=over, cmap='viridis', s=50)
    ax.set_xlabel('Coverage Ratio')
    ax.set_ylabel('Avg RSSI (dBm)')
    ax.set_zlabel('Overlap Metric')
    plt.colorbar(sc, label='Overlap')
    plt.title('Approximated Pareto Front (MOEA/D)')
    plt.savefig('moea-d/pareto_front.png')
    print("Saved pareto_front.png")
    
    # 2. Final Layout Plots
    # Pick a "Balanced" solution (e.g., middle of population sorted by one objective)
    # Or best for each objective.
    
    def get_best_idx(obj_idx):
        return np.argmin(all_objs[:, obj_idx])
        
    # Best Coverage (Min -Cov)
    plot_layout(final_pop[get_best_idx(0)], "Best Coverage", "layout_best_coverage.png")
    
    # Best Quality (Min -RSSI)
    plot_layout(final_pop[get_best_idx(1)], "Best Quality", "layout_best_quality.png")
    
    # Best Overlap (Min Overlap)
    plot_layout(final_pop[get_best_idx(2)], "Minimum Overlap", "layout_min_overlap.png")

    # Balanced Solution (Median of Coverage)
    # Sort by coverage (descending, since objective is -Coverage)
    # objective[0] is -Coverage. Smallest is best (e.g. -0.9).
    # Sorting by objective[0] ascending gives best coverage first (-0.95, -0.9, ... -0.5).
    sorted_by_cov = sorted(final_pop, key=lambda x: x.objectives[0])
    idx_bal = len(sorted_by_cov) // 2
    plot_layout(sorted_by_cov[idx_bal], "Balanced Trade-off", "layout_balanced.png")

def plot_layout(individual, title_prefix, filename):
    routers = individual.routes
    # Create a fresh env to access static map info and calc RSSI for heatmap
    env = MOEADSolver().env 
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- HEATMAP ---
    resolution = 0.5 
    gx = np.arange(0, GRID_SIZE, resolution)
    gy = np.arange(0, GRID_SIZE, resolution)
    grid_x, grid_y = np.meshgrid(gx, gy)
    flat_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # Calculate Max RSSI at each point
    # We want visual heatmap of signal strength
    n_points = len(flat_points)
    all_rssi = np.zeros((len(routers), n_points))
    
    for i, router in enumerate(routers):
        all_rssi[i, :] = env.calculate_rssi_vectorized(router, flat_points)
        
    max_rssi = np.max(all_rssi, axis=0)
    rssi_map = max_rssi.reshape(grid_x.shape)
    
    # Plot Heatmap
    im = ax.imshow(rssi_map, extent=[0, GRID_SIZE, 0, GRID_SIZE], origin='lower', cmap='RdYlGn', alpha=0.6, vmin=-90, vmax=-40)
    plt.colorbar(im, label='Signal Strength (dBm)')
    
    # Obstacles
    for obs in env.obstacles:
        x, y, w, h = obs
        rect = plt.Rectangle((x, y), w, h, color='gray', alpha=0.5, label='_nolegend_')
        ax.add_patch(rect)
        
    # Clients
    clients = env.clients
    ax.scatter(clients[:, 0], clients[:, 1], c='blue', marker='o', s=50, label='Clients', edgecolors='white')
    
    # Routers
    ax.scatter(routers[:, 0], routers[:, 1], c='red', marker='*', s=300, label='Routers', edgecolors='white')
    
    # Circles
    for r in routers:
        circle = plt.Circle((r[0], r[1]), 31.6, color='red', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(circle)
        
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    # Title
    cov = -individual.objectives[0] * 100
    rssi = -individual.objectives[1]
    over = individual.objectives[2]
    
    plt.title(f'{title_prefix}\nCov: {cov:.1f}%, RSSI: {rssi:.1f} dBm, Overlap: {over:.1f}')
    plt.savefig(f'moea-d/{filename}')
    print(f"Saved {filename}")

if __name__ == "__main__":
    main()
