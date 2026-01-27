
import numpy as np

# Grid Settings
GRID_SIZE = 100
GRID_RESOLUTION = 1.0  # meter per unit (implied)

# Environment Settings
N_ROUTERS = 5
N_CLIENTS = 20
OBSTACLE_ATTENUATION = 15.0  # dB loss per wall crossed
# Log-Distance Path Loss Model Parameters
P_TX = 20.0  # Transmit power in dBm
L0 = 40.0    # Reference path loss at 1m (in dB)
GAMMA = 3.0  # Path loss exponent (urban/indoor environment)

# Optimization constraints
MIN_RSSI_COVERAGE = -80.0  # dBm
MIN_RSSI_OVERLAP = -70.0   # dBm

# MOEA/D Settings
POPULATION_SIZE = 100 # Adjusted for weight vectors
GENERATIONS = 1000
NEIGHBORHOOD_SIZE = 10 # T
MUTATION_RATE = 0.1

# Random Seed
SEED = 42
