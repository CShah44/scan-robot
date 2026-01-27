# Prompt: NSGA-II Implementation for Optimal Router Placement

**Role:** Expert in Evolutionary Computation, Robotics, and Wireless Signal Modeling.

**Objective:** Develop a Python script using `numpy` and `matplotlib` that solves a Multi-Objective Router Placement problem. The implementation must strictly follow the mathematical framework of the **NSGA-II** algorithm (Deb et al., 2002) to optimize router coordinates within a predefined grid layout containing obstacles and fixed clients.

---

### 1. Environment & Problem Constraints
* **The Grid:** A $100 \times 100$ NumPy matrix representing the floor plan.
* **Obstacles:** Define a set of rectangular regions (coordinates) where routers cannot be placed and which attenuate signal (walls).
* **Clients:** 20 fixed $(x, y)$ coordinates representing users requiring Wi-Fi.
* **Routers:** $N = 5$ routers to be placed. Each "Individual" in the population is an array of $5 \times 2$ coordinates.

### 2. Strict NSGA-II Algorithmic Requirements
The script must implement the following from the original paper:
* **Fast Non-Dominated Sorting:** Use the $O(MN^2)$ approach. Maintain domination count ($n_p$) and the set of dominated individuals ($S_p$).
* **Crowding Distance Assignment:** For each front, calculate density by sorting each objective.
    * Formula: $I[i]_{dist} = I[i]_{dist} + \frac{f_m(i+1) - f_m(i-1)}{f_m^{max} - f_m^{min}}$.
    * Boundary points must be assigned $\infty$ distance.
* **Selection:** Binary Tournament Selection using the **Crowded-Comparison Operator** ($\prec_n$): Prefer lower rank; if ranks are equal, prefer higher crowding distance.
* **Elitism:** Combine parent ($P_t$) and offspring ($Q_t$) populations ($2N$) before selecting the best $N$ for the next generation.
* **Variation:** Implement **Simulated Binary Crossover (SBX)** and **Polynomial Mutation**.

### 3. Objective Functions (Fitness Calculations)
Implement three conflicting objectives:
1.  **Maximize Coverage ($f_1$):** Percentage of clients with $RSSI > -80\text{ dBm}$. 
2.  **Maximize Signal Quality ($f_2$):** Average RSSI across all clients. Use the Log-Distance Path Loss Model:
    $$RSSI = P_{tx} - (L_0 + 10\gamma \log_{10}(d) + \sum N_{walls} \times L_{wall})$$
    *(Include a basic Ray-Casting/Line-of-Sight check to count wall intersections).*
3.  **Minimize Overlap ($f_3$):** Total grid area where more than one router provides signal $> -70\text{ dBm}$.

### 4. Visualization & Output
* **Final Layout Plot:** Display the $100 \times 100$ grid with obstacles (gray), clients (blue dots), and routers (red stars) with colored circles representing signal range.
* **Pareto Front Plot:** A 3D scatter plot showing the trade-offs between Coverage, RSSI, and Overlap for the final population.
* **Convergence Plot:** Average fitness/rank over generations.

### 5. Code Structure
* Modular classes: `Environment`, `Individual`, and `NSGA2Solver`.
* Parameters: Population Size = 50, Generations = 100, Mutation Rate = 0.1.