import numpy as np
import copy
from .config import *
from .environment import Environment

class Individual:
    def __init__(self, routes=None):
        if routes is None:
            # Random initialization within grid bounds
            self.routes = np.random.uniform(0, GRID_SIZE, size=(N_ROUTERS, 2))
        else:
            self.routes = routes
        
        # Objectives: [Minimize (-Coverage), Minimize (-Quality), Minimize (Overlap)]
        self.objectives = None 
        self.rank = -1
        self.crowding_distance = 0.0
        
    def evaluate(self, env):
        self.objectives = env.evaluate_full(self.routes)
        
    def dominates(self, other):
        """
        Returns True if self dominates other.
        Domination: <= in all objectives AND < in at least one.
        """
        if self.objectives is None or other.objectives is None:
            return False
            
        obj1 = np.array(self.objectives)
        obj2 = np.array(other.objectives)
        
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

class NSGA2Solver:
    def __init__(self):
        self.env = Environment(seed=SEED)
        self.population = []
        
    def initialize_population(self):
        print("Initializing population...")
        self.population = []
        for _ in range(POPULATION_SIZE):
            ind = Individual()
            ind.evaluate(self.env)
            self.population.append(ind)
            
    def fast_non_dominated_sort(self, population):
        fronts = [[]]
        for p in population:
            p.S_p = [] # Individuals that p dominates
            p.n_p = 0  # Number of individuals that dominate p
            
            for q in population:
                if p.dominates(q):
                    p.S_p.append(q)
                elif q.dominates(p):
                    p.n_p += 1
            
            if p.n_p == 0:
                p.rank = 0
                fronts[0].append(p)
                
        i = 0
        while len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in p.S_p:
                    q.n_p -= 1
                    if q.n_p == 0:
                        q.rank = i + 1
                        Q.append(q)
            i += 1
            if len(Q) > 0:
                fronts.append(Q)
            else:
                break
                
        # Remove the last empty front if it exists
        if len(fronts) > 0 and len(fronts[-1]) == 0:
            fronts.pop()
            
        return fronts

    def calculate_crowding_distance(self, front):
        length = len(front)
        if length == 0:
            return
        
        for p in front:
            p.crowding_distance = 0.0
            
        n_obj = len(front[0].objectives)
        
        for m in range(n_obj):
            # Sort by objective m
            front.sort(key=lambda x: x.objectives[m])
            
            # Boundary points get infinity distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            obj_min = front[0].objectives[m]
            obj_max = front[-1].objectives[m]
            
            if obj_max - obj_min == 0:
                continue
                
            scale = obj_max - obj_min
            for i in range(1, length - 1):
                if front[i].crowding_distance != float('inf'):
                    dist = (front[i+1].objectives[m] - front[i-1].objectives[m]) / scale
                    front[i].crowding_distance += dist

    def crowded_comparison(self, ind1, ind2):
        if ind1.rank < ind2.rank:
            return ind1
        elif ind1.rank > ind2.rank:
            return ind2
        else:
            if ind1.crowding_distance > ind2.crowding_distance:
                return ind1
            else:
                return ind2

    def binary_tournament_selection(self, population):
        # We need to return POPULATION_SIZE parents
        mating_pool = []
        while len(mating_pool) < len(population):
            # Pick two random individuals
            candidates = np.random.choice(population, size=2, replace=False)
            winner = self.crowded_comparison(candidates[0], candidates[1])
            mating_pool.append(winner)
        return mating_pool

    def sbx_crossover(self, p1, p2):
        """Simulated Binary Crossover"""
        # Crossover probability fixed high usually, or 0.9
        # Assuming we just do it for variables
        
        off1_routes = np.zeros_like(p1.routes)
        off2_routes = np.zeros_like(p2.routes)
        
        for i in range(p1.routes.size):
            val1 = p1.routes.flat[i]
            val2 = p2.routes.flat[i]
            
            if np.random.rand() <= 0.5:
                # Swap or mix
                if np.abs(val1 - val2) > 1e-6:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (ETA_C + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (ETA_C + 1.0))
                    
                    c1 = 0.5 * ((1 + beta) * val1 + (1 - beta) * val2)
                    c2 = 0.5 * ((1 - beta) * val1 + (1 + beta) * val2)
                else:
                    c1, c2 = val1, val2
            else:
                 c1, c2 = val1, val2
            
            # Clip to bounds
            c1 = np.clip(c1, 0, GRID_SIZE)
            c2 = np.clip(c2, 0, GRID_SIZE)
            
            off1_routes.flat[i] = c1
            off2_routes.flat[i] = c2
            
        return Individual(off1_routes), Individual(off2_routes)

    def polynomial_mutation(self, ind):
        """Polynomial Mutation"""
        mutated_routes = np.copy(ind.routes)
        
        for i in range(mutated_routes.size):
            if np.random.rand() < MUTATION_RATE:
                val = mutated_routes.flat[i]
                low, high = 0.0, float(GRID_SIZE)
                delta_range = high - low
                
                u = np.random.rand()
                # Delta calculation
                # Simplified polynomial mutation
                rk = (val - low) / delta_range
                
                if u <= 0.5:
                    delta_q = (2 * u + (1 - 2 * u) * (1 - rk) ** (ETA_M + 1)) ** (1 / (ETA_M + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - rk) ** (ETA_M + 1)) ** (1 / (ETA_M + 1))
                
                new_val = val + delta_q * delta_range
                new_val = np.clip(new_val, low, high)
                mutated_routes.flat[i] = new_val
                
        return Individual(mutated_routes)

    def create_offspring(self, population):
        # Selection
        parents = self.binary_tournament_selection(population)
        offspring_pop = []
        
        # Crossover & Mutation
        for i in range(0, len(parents), 2):
            if i+1 < len(parents):
                p1 = parents[i]
                p2 = parents[i+1]
                # SBX
                c1, c2 = self.sbx_crossover(p1, p2)
                
                # Mutation
                c1 = self.polynomial_mutation(c1)
                c2 = self.polynomial_mutation(c2)
                
                c1.evaluate(self.env)
                c2.evaluate(self.env)
                
                offspring_pop.append(c1)
                offspring_pop.append(c2)
        
        return offspring_pop

    def solve(self):
        self.initialize_population()
        self.fast_non_dominated_sort(self.population)
        # Assign crowding distance
        fronts = self.fast_non_dominated_sort(self.population)
        for front in fronts:
            self.calculate_crowding_distance(front)
            
        params_convergence = []
        
        for gen in range(GENERATIONS):
            print(f"Generation {gen+1}/{GENERATIONS}")
            
            # Create offspring
            offspring = self.create_offspring(self.population)
            
            # Combine
            R_t = self.population + offspring
            
            # Non-dominated sort combined population
            fronts = self.fast_non_dominated_sort(R_t)
            
            next_population = []
            
            # Fill next population
            for front in fronts:
                self.calculate_crowding_distance(front)
                if len(next_population) + len(front) <= POPULATION_SIZE:
                    next_population.extend(front)
                else:
                    # Sort by crowding distance (descending) and fill remaining
                    # Crowding distance comparison: higher is better
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    next_population.extend(front[:POPULATION_SIZE - len(next_population)])
                    break
            
            self.population = next_population
            
            # Log progress
            # Average rank? Or just track best fitness?
            # Let's track average objectives of rank 0
            rank0 = [ind for ind in self.population if ind.rank == 0]
            if rank0:
                avg_obj = np.mean([ind.objectives for ind in rank0], axis=0)
                params_convergence.append(avg_obj)
            else:
                 # Should not happen
                 params_convergence.append([0,0,0])

        return self.population, params_convergence

