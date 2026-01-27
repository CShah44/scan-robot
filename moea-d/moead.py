
import numpy as np
import copy
from .config import *
from .environment import Environment

class Individual:
    def __init__(self, routes=None):
        if routes is None:
            # Uniform random init
            self.routes = np.random.uniform(0, GRID_SIZE, size=(N_ROUTERS, 2))
        else:
            self.routes = routes
        
        self.objectives = None # [f1, f2, f3]
        
    def evaluate(self, env):
        self.objectives = np.array(env.evaluate(self.routes))

class MOEADSolver:
    def __init__(self):
        self.env = Environment(seed=SEED)
        self.population = []
        self.weights = []
        self.z_ideal = None
        self.neighborhoods = [] # Indices of T nearest weights
        
    def init_weights(self):
        """
        Generate uniformly distributed weight vectors.
        For 3 objectives, simplest valid approach matching POPULATION_SIZE
        is uniform random sampling + normalization.
        """
        # Option: Simplex Lattice Design if we want perfect uniformity
        # But to respect POPULATION_SIZE exactly, we use random.
        # Although "Instructions.md" mentions simplex-lattice, it allows uniform random.
        
        # Using Uniform Random Sampling
        # Sample on simplex: Dirichelt distribution with alpha=1 gives uniform on simplex
        self.weights = np.random.dirichlet(np.ones(3), size=POPULATION_SIZE)
        
        # Ensure minimal weight to avoid 0 division potential (though not strictly needed for Tchebycheff)
        self.weights = np.maximum(self.weights, 1e-6)
        
    def init_neighborhoods(self):
        """Calculate Euclidean distance matrix of weights and find T neighbors."""
        from scipy.spatial.distance import cdist
        dists = cdist(self.weights, self.weights)
        
        self.neighborhoods = []
        for i in range(POPULATION_SIZE):
            # Argsort and take T closest (including self)
            neighbors = np.argsort(dists[i])[:NEIGHBORHOOD_SIZE]
            self.neighborhoods.append(neighbors)
            
    def initialize(self):
        self.init_weights()
        self.init_neighborhoods()
        
        self.population = []
        for _ in range(POPULATION_SIZE):
            ind = Individual()
            ind.evaluate(self.env)
            self.population.append(ind)
            
        # Initialize Ideal Point z*
        all_objs = np.array([ind.objectives for ind in self.population])
        # Min of minimization objectives
        self.z_ideal = np.min(all_objs, axis=0)
        
    def tchebycheff(self, individual, weight_vec):
        """
        g(x | lambda, z*) = max_i ( lambda_i * | f_i(x) - z*_i | )
        """
        diff = np.abs(individual.objectives - self.z_ideal)
        max_val = np.max(weight_vec * diff)
        return max_val
    
    def crossover(self, p1, p2):
        """
        Uniform crossover at router level.
        Router positions inherited independently.
        """
        off_routes = np.zeros_like(p1.routes)
        for i in range(N_ROUTERS):
            if np.random.rand() < 0.5:
                off_routes[i] = p1.routes[i]
            else:
                off_routes[i] = p2.routes[i]
        
        return Individual(off_routes)
    
    def mutation(self, ind):
        """
        Move router by +/- 1 grid cell OR re-sample.
        """
        mut_routes = np.copy(ind.routes)
        
        for i in range(N_ROUTERS):
            if np.random.rand() < MUTATION_RATE:
                if np.random.rand() < 0.5:
                    # Perturb +/- 1
                    dx = np.random.uniform(-1, 1)
                    dy = np.random.uniform(-1, 1)
                    mut_routes[i] += [dx, dy]
                else:
                    # Resample
                    mut_routes[i] = np.random.uniform(0, GRID_SIZE, 2)
                    
                # Clip
                mut_routes[i] = np.clip(mut_routes[i], 0, GRID_SIZE)
                
        return Individual(mut_routes)
    
    def solve(self):
        print("Initializing MOEA/D...")
        self.initialize()
        
        for gen in range(GENERATIONS):
            print(f"Generation {gen+1}/{GENERATIONS}")
            
            for i in range(POPULATION_SIZE):
                # 1. Selection: Randomly select 2 parents from Neighborhood
                # Neighborhood B(i)
                B_i = self.neighborhoods[i]
                
                # Pick 2 distinct indices from B_i
                p_indices = np.random.choice(B_i, 2, replace=False)
                p1 = self.population[p_indices[0]]
                p2 = self.population[p_indices[1]]
                
                # 2. Reproduction
                offspring = self.crossover(p1, p2)
                offspring = self.mutation(offspring)
                offspring.evaluate(self.env)
                
                # 3. Update Ideal Point
                self.z_ideal = np.minimum(self.z_ideal, offspring.objectives)
                
                # 4. Update Neighbors
                for j in B_i:
                    g_off = self.tchebycheff(offspring, self.weights[j])
                    g_curr = self.tchebycheff(self.population[j], self.weights[j])
                    
                    if g_off <= g_curr:
                        self.population[j] = offspring
                        
        return self.population
