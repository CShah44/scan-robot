
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re

# Add parent directory to path to allow importing from reoptimization package if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from reoptimization.nsga2 import NSGA2Solver
from reoptimization.config import *
from reoptimization.main import plot_layout, log_results


## Input
LOG_FILE = os.path.join(current_dir, 'simulation_log_100.txt')

def parse_log(filename):
    """
    Parses the simulation log to extract the 'Best Coverage Solution' router coordinates.
    Returns a list of (x, y) tuples.
    """
    routers = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        in_section = False
        for line in lines:
            if "--- Best Coverage Solution ---" in line:
                in_section = True
                continue
            
            if in_section and "Router Coordinates:" in line:
                continue
                
            if in_section and line.strip().startswith("Router"):
                # matches "  Router 1: (53.16, 97.93)"
                match = re.search(r"Router \d+: \(([\d\.]+), ([\d\.]+)\)", line)
                if match:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    routers.append((x, y))
            
            if in_section and line.strip() == "" and len(routers) > 0:
                break
                
        return routers
    except FileNotFoundError:
        print(f"Error: Log file not found at {filename}")
        return []

def plot_initial_state(routers):
    """
    Plots the initial router positions for the user to see.
    """
    routers_arr = np.array(routers)
    
    # Create a dummy env to get clients and obstacles
    solver = NSGA2Solver()
    env = solver.env
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Obstacles
    for obs in env.obstacles:
        x, y, w, h = obs
        rect = plt.Rectangle((x, y), w, h, color='gray', alpha=0.5)
        ax.add_patch(rect)
        
    # Clients
    clients = env.clients
    ax.scatter(clients[:, 0], clients[:, 1], c='blue', marker='o', label='Clients')
    
    # Routers
    for i, (rx, ry) in enumerate(routers):
        ax.scatter(rx, ry, c='red', marker='*', s=200, edgecolors='black')
        ax.text(rx+1, ry+1, f"R{i+1}", fontsize=12, fontweight='bold', color='darkred')
        # Range circle
        circle = plt.Circle((rx, ry), 31.6, color='red', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(circle)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Current Router Positions (Best Coverage)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    print("Displaying initial state plot...")
    plt.show(block=False) 
    # block=False so script continues to input. 
    # BUT standard input might block. We usually pause.
    plt.pause(0.1)

def reoptimize(failed_router_indices, original_routers):
    """
    Re-optimize router placement after removing failed routers.
    """
    original_n = len(original_routers)
    n_new_routers = original_n - len(failed_router_indices)
    
    if n_new_routers <= 0:
        print("Error: No routers left to place!")
        return

    print(f"\n--- Re-optimizing for {n_new_routers} routers (Original: {original_n}, Failed: {len(failed_router_indices)}) ---")
    
    solver = NSGA2Solver(n_routers=n_new_routers)
    final_pop, conv_hist = solver.solve()
    print("Optimization Completed.")
    
    # --- Visualization & Logging ---
    rank0_pop = [ind for ind in final_pop if ind.rank == 0]
    
    if not rank0_pop:
        print("No rank 0 solutions found!")
        return
        
    def get_best_idx(objective_idx, minimize=True):
        vals = [ind.objectives[objective_idx] for ind in rank0_pop]
        return np.argmin(vals) if minimize else np.argmax(vals)

    # Best Coverage
    idx_cov = get_best_idx(0, minimize=True)
    plot_layout(rank0_pop[idx_cov], f"Best Coverage ({n_new_routers} Routers)", "layout_reopt_coverage.png")
    
    # Log results
    log_file = os.path.join(current_dir, 'simulation_log_reoptimized.txt')
    log_results(log_file, solver, final_pop)
    print(f"Re-optimization details logged to {log_file}")
    
    # Show the result plot
    result_img = os.path.join(current_dir, "layout_reopt_coverage.png")
    # print(f"Result image saved to: {result_img}")

    # --- Plot Comparison ---
    try:
        best_cov_ind = rank0_pop[idx_cov]
        new_routers = best_cov_ind.routes
        plot_comparison(original_routers, failed_router_indices, new_routers)
    except Exception as e:
        print(f"Error plotting comparison: {e}")

def plot_comparison(original_routers, failed_indices, new_routers):
    """
    Plots a side-by-side comparison of Before (Failure) vs After (Re-optimization).
    """
    original_routers = np.array(original_routers)
    new_routers = np.array(new_routers)
    
    # Create dummy env for background
    solver = NSGA2Solver()
    env = solver.env
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- Subplot 1: Before (with Failures) ---
    ax = axes[0]
    ax.set_title("Before: Router Failure")
    
    # Background
    for obs in env.obstacles:
        x, y, w, h = obs
        ax.add_patch(plt.Rectangle((x, y), w, h, color='gray', alpha=0.5))
    ax.scatter(env.clients[:, 0], env.clients[:, 1], c='blue', marker='o', label='Clients')
    
    # Plot Routers
    for i, (rx, ry) in enumerate(original_routers):
        if i in failed_indices:
            # FAILED: Black X
            ax.scatter(rx, ry, c='black', marker='x', s=300, linewidths=3, label='Failed' if i == failed_indices[0] else "")
            ax.text(rx+2, ry+2, f"R{i+1}(Fail)", fontsize=10, color='black', fontweight='bold')
        else:
            # ACTIVE: Red Star
            ax.scatter(rx, ry, c='red', marker='*', s=200, edgecolors='black', label='Active' if i == 0 else "")
            ax.text(rx+2, ry+2, f"R{i+1}", fontsize=10, color='darkred')
            # Range
            ax.add_patch(plt.Circle((rx, ry), 31.6, color='red', fill=False, linestyle='--', alpha=0.3))
            
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='upper right')

    # --- Subplot 2: After (Re-optimized) ---
    ax = axes[1]
    n_new = len(new_routers)
    ax.set_title(f"After: Re-optimized ({n_new} Routers)")
    
    # Background
    for obs in env.obstacles:
        x, y, w, h = obs
        ax.add_patch(plt.Rectangle((x, y), w, h, color='gray', alpha=0.5))
    ax.scatter(env.clients[:, 0], env.clients[:, 1], c='blue', marker='o')
    
    # Plot New Routers
    for i, (rx, ry) in enumerate(new_routers):
        ax.scatter(rx, ry, c='green', marker='*', s=300, edgecolors='black', label='New Router' if i == 0 else "")
        ax.text(rx+2, ry+2, f"N{i+1}", fontsize=10, color='darkgreen', fontweight='bold')
        # Range
        ax.add_patch(plt.Circle((rx, ry), 31.6, color='green', fill=False, linestyle='--', alpha=0.3))
        
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    save_path = os.path.join(current_dir, "comparison_reopt.png")
    plt.savefig(save_path)
    print(f"Comparison image saved to: {save_path}")
    plt.show()

def main():
    print("--- Router Failure Recovery System ---")
    
    # 1. Parse Logic
    routers = parse_log(LOG_FILE)
    if not routers:
        print("Could not find initial router positions. Using default N=5 for display.")
        routers = [(50,50)] * 5 # Dummy
    
    print(f"Found {len(routers)} active routers from log.")
    for i, r in enumerate(routers):
        print(f"  Router {i+1}: ({r[0]:.2f}, {r[1]:.2f})")
        
    # 2. Visualize
    try:
        plot_initial_state(routers)
    except Exception as e:
        print(f"Warning: Could not plot initial state: {e}")

    # 3. User Input
    print("\nSelect failed routers.")
    print("Enter the ID(s) of the routers that failed (e.g., '1 3' for Router 1 and Router 3).")
    
    if len(sys.argv) > 1:
        # Debug/CLI mode
        input_str = " ".join(sys.argv[1:])
        print(f"Auto-input from CLI: {input_str}")
    else:
        try:
            input_str = input("Failed Router IDs > ")
        except EOFError:
            print("No input provided.")
            return

    try:
        # Convert to 0-based indices
        failed_indices = [int(x.strip()) - 1 for x in input_str.split() if x.strip().isdigit()]
        
        # Validate
        valid_indices = []
        for idx in failed_indices:
            if 0 <= idx < len(routers):
                valid_indices.append(idx)
            else:
                print(f"Warning: Router ID {idx+1} is out of range. Ignoring.")
        
        if not valid_indices:
            print("No new failed routers specified. Exiting.")
            return
            
        print(f"Marked as failed: {[i+1 for i in valid_indices]}")
        reoptimize(valid_indices, routers)
        
    except ValueError:
        print("Invalid input. Please enter numbers.")

if __name__ == "__main__":
    main()
