import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .nsga2 import NSGA2Solver
from .config import *
import os

def main():
    print("Starting NSGA-II Optimization...")
    solver = NSGA2Solver()
    final_pop, conv_hist = solver.solve()
    print("Optimization Completed.")
    
    # Ensure plots directory exists? Or just save in current dir?
    
    # 1. Pareto Front Plot (3D)
    pareto_front = [ind.objectives for ind in final_pop if ind.rank == 0]
    pareto_front = np.array(pareto_front)
    
    if len(pareto_front) > 0:
        # Objectives are minimized: [-Coverage, -Quality, Overlap]
        # Plot: Coverage, Quality, Overlap
        cov = -pareto_front[:, 0]
        qual = -pareto_front[:, 1]
        over = pareto_front[:, 2]
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(cov, qual, over, c=over, cmap='viridis', s=50)
        ax.set_xlabel('Coverage Ratio')
        ax.set_ylabel('Avg RSSI (dBm)')
        ax.set_zlabel('Overlap Area')
        plt.colorbar(sc, label='Overlap')
        plt.title('Pareto Front')
        plt.savefig('nsga2_router_placement/pareto_front.png')
        print("Saved pareto_front.png")
    
    # 2. Convergence Plot
    if len(conv_hist) > 0:
        conv_hist = np.array(conv_hist)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(-conv_hist[:, 0])
        axes[0].set_title('Avg Coverage (Rank 0)')
        axes[0].set_xlabel('Generation')
        
        axes[1].plot(-conv_hist[:, 1])
        axes[1].set_title('Avg Signal Quality (Rank 0)')
        axes[1].set_xlabel('Generation')
        
        axes[2].plot(conv_hist[:, 2])
        axes[2].set_title('Avg Overlap (Rank 0)')
        axes[2].set_xlabel('Generation')
        
        plt.tight_layout()
        plt.savefig('nsga2_router_placement/convergence.png')
        print("Saved convergence.png")
    
    # 3. Final Layout Plots
    if len(pareto_front) > 0:
        rank0_pop = [ind for ind in final_pop if ind.rank == 0]
        
        # Helper to find index
        def get_best_idx(objective_idx, minimize=True):
            vals = [ind.objectives[objective_idx] for ind in rank0_pop]
            if minimize:
                return np.argmin(vals)
            else:
                return np.argmax(vals)

        # A. Best Coverage (Min of -Coverage)
        idx_cov = get_best_idx(0, minimize=True)
        plot_layout(rank0_pop[idx_cov], "Best Coverage", "layout_best_coverage.png")
        
        # B. Best Signal Quality (Min of -Avg RSSI)
        idx_qual = get_best_idx(1, minimize=True)
        plot_layout(rank0_pop[idx_qual], "Best Signal Quality", "layout_best_quality.png")
        
        # C. Minimum Overlap (Min of Overlap)
        idx_over = get_best_idx(2, minimize=True)
        plot_layout(rank0_pop[idx_over], "Minimum Overlap", "layout_min_overlap.png")
        
        # D. Balanced Solution (Median of Coverage)
        # Sort by coverage
        rank0_pop.sort(key=lambda x: x.objectives[0])
        idx_bal = len(rank0_pop) // 2
        plot_layout(rank0_pop[idx_bal], "Balanced Trade-off", "layout_balanced.png")

def plot_layout(individual, title_prefix, filename):
    routers = individual.routes
    # Create a dummy env to access static map info
    env = NSGA2Solver().env 
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- HEATMAP ---
    # Generate grid for heatmap
    resolution = 0.5 # 0.5 meter resolution for smoother look
    gx = np.arange(0, GRID_SIZE, resolution)
    gy = np.arange(0, GRID_SIZE, resolution)
    grid_x, grid_y = np.meshgrid(gx, gy)
    flat_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # Calculate Max RSSI at each point
    num_points = flat_points.shape[0]
    all_rssi = np.zeros((len(routers), num_points))
    
    for i, router in enumerate(routers):
        all_rssi[i, :] = env.calculate_rssi_vectorized(router, flat_points)
        
    max_rssi = np.max(all_rssi, axis=0)
    rssi_map = max_rssi.reshape(grid_x.shape)
    
    # Plot Heatmap
    # vmin/vmax tailored to typical WiFi ranges (-90 to -30)
    im = ax.imshow(rssi_map, extent=[0, GRID_SIZE, 0, GRID_SIZE], origin='lower', cmap='RdYlGn', alpha=0.6, vmin=-90, vmax=-40)
    plt.colorbar(im, label='Signal Strength (dBm)')
    # ---------------

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
    
    # Signal range circles (visual aid)
    # Range with 0 walls ~ 100m, 1 wall ~ 31.6m, 2 walls ~ 10m
    # We plot the 1-wall range (31.6m) as a reference.
    for r in routers:
        # Drawing ~32m circle (1-wall limit)
        circle = plt.Circle((r[0], r[1]), 31.6, color='red', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(circle)
        
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3)
    plt.legend()
    # Format objectives for title
    cov = -individual.objectives[0] * 100
    rssi = -individual.objectives[1]
    over = individual.objectives[2]
    
    plt.title(f'{title_prefix}\nCov: {cov:.1f}%, RSSI: {rssi:.1f} dBm, Overlap: {over:.0f} m^2')
    plt.savefig(f'nsga2_router_placement/{filename}')
    print(f"Saved {filename}")

if __name__ == "__main__":
    main()
