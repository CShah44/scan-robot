# NSGA-II Router Placement Optimization: Mathematical Models

This document details the mathematical functions, physical models, and evolutionary algorithms used in the router placement optimization code.

## 1. Physics & Signal Propagation (Environment)

These functions define the simulation environment in `environment.py`, specifically how signal strength is calculated and how obstacles affect propagation.

### Log-Distance Path Loss Model
Used to calculate the **Received Signal Strength Indicator (RSSI)**.

$$ RSSI(d) = P_{tx} - PL(d) $$

$$ PL(d) = L_0 + 10 \cdot \gamma \cdot \log_{10}(d) + N_{walls} \cdot \alpha $$

**Definitions:**
*   **$P_{tx}$ (Transmit Power):** `20.0 dBm`
*   **$L_0$ (Reference Path Loss at 1m):** `40.0 dB`
*   **$\gamma$ (Gamma, Path Loss Exponent):** `3.0` (Urban/Indoor environment)
*   **$\alpha$ (Obstacle Attenuation):** `15.0 dB` per wall
*   **$d$:** Euclidean distance between the router and the target point.
*   **$N_{walls}$:** Number of obstacles intersected by the line of sight.

### Line Segment Intersection
Used to calculate $N_{walls}$ by performing ray casting.
Given a ray from Router ($P_1$) to Client ($P_2$) and a wall segment ($P_3 \to P_4$):

$$ d = (y_4 - y_3)(x_2 - x_1) - (x_4 - x_3)(y_2 - y_1) $$

Intersection parameters $u_a, u_b$ are calculated as:

$$ u_a = \frac{(x_4 - x_3)(y_1 - y_3) - (y_4 - y_3)(x_1 - x_3)}{d} $$

$$ u_b = \frac{(x_2 - x_1)(y_1 - y_3) - (y_2 - y_1)(x_1 - x_3)}{d} $$

An intersection occurs if $0 \le u_a \le 1$ and $0 \le u_b \le 1$.

---

## 2. Optimization Objectives

The genetic algorithm minimizes three cost functions. Since the goals are naturally maximization (Coverage, Quality), they are inverted (multiplied by -1).

### Objective 1: Maximize Coverage Ratio
Minimizes the negative percentage of clients with adequate signal.

$$ f_1 = - \frac{1}{N_{clients}} \sum_{i=1}^{N_{clients}} \mathbb{I}(\max_{j} RSSI_{i,j} \ge -80 \text{ dBm}) $$

*   **Goal:** Ensure as many clients as possible have at least one router with signal $\ge$ -80 dBm.

### Objective 2: Maximize Signal Quality (Avg RSSI)
Minimizes the negative average signal strength.

$$ f_2 = - \frac{1}{N_{clients}} \sum_{i=1}^{N_{clients}} (\max_{j} RSSI_{i,j}) $$

*   **Goal:** maximize the average signal strength received by clients (not just barely connected, but strong connection).

### Objective 3: Minimize Overlap Area
Minimizes the physical area where signals interfere.

$$ f_3 = Area_{total} \times \frac{1}{N_{grid}} \sum_{k=1}^{N_{grid}} \mathbb{I} \left( (\sum_{j} \mathbb{I}(RSSI_{k,j} \ge -70 \text{ dBm})) > 1 \right) $$

*   **Goal:** Reduce area where multiple routers provide strong signal ($\ge$ -70 dBm) to save power and reduce interference.

---

## 3. Evolutionary Operators (NSGA-II)

These functions drive the genetic search in `nsga2.py`.

### Pareto Domination
Solution A dominates Solution B ($A \prec B$) if:
1.  $\forall i, f_i(A) \le f_i(B)$ (A is no worse than B in all objectives)
2.  $\exists j, f_j(A) < f_j(B)$ (A is strictly better than B in at least one objective)

### Crowding Distance
Used to maintain diversity in the population.

$$ distance_i = \sum_{m=1}^{M} \frac{f_m(i+1) - f_m(i-1)}{f_m^{max} - f_m^{min}} $$

*   Solutions in less crowded regions of the objective space are preferred during selection.

### Simulated Binary Crossover (SBX)
Generates offspring $c_1, c_2$ from parents $p_1, p_2$.
Spread factor $\beta$ is calculated using a random number $u \in [0,1]$ and distribution index $\eta_c = 20$:

$$ \beta = \begin{cases} (2u)^{\frac{1}{\eta_c+1}} & \text{if } u \le 0.5 \\ (\frac{1}{2(1-u)})^{\frac{1}{\eta_c+1}} & \text{if } u > 0.5 \end{cases} $$

$$ c_{1} = 0.5 \cdot [(1+\beta)p_{1} + (1-\beta)p_{2}] $$
$$ c_{2} = 0.5 \cdot [(1-\beta)p_{1} + (1+\beta)p_{2}] $$

### Polynomial Mutation
Perturbs a gene $x$ with a shift calculated from random number $u \in [0,1]$ and index $\eta_m = 20$.

$$ \delta_q = \begin{cases} [2u + (1-2u)(1-\delta)^{\eta_m+1}]^{\frac{1}{\eta_m+1}} - 1 & \text{if } u \le 0.5 \\ 1 - [2(1-u) + 2(u-0.5)(1-\delta)^{\eta_m+1}]^{\frac{1}{\eta_m+1}} & \text{if } u > 0.5 \end{cases} $$

Where $\delta$ represents the normalized distance of the current value to the boundary.
