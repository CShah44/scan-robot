# Task: Implement MOEA/D for Optimal Wi-Fi Router Placement (Python)

You are to implement a **Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D)** to solve a **Wi-Fi router placement optimization problem** in a grid-based environment with obstacles and clients.

The implementation must be **correct, modular, and visualizable**, using **Python + NumPy + Matplotlib**.

---

## 1. Problem Definition

### 1.1 Environment

- 2D grid of size `(W × H)`
- Discrete grid cells
- Some cells are **obstacles** (signal attenuating or blocking)
- Some cells contain **clients**
- We place **n routers** at valid (non-obstacle) grid cells

---

### 1.2 Decision Variables

A solution (individual) is defined as:

\[
\mathbf{x} = [x_1, y_1, x_2, y_2, \dots, x_n, y_n]
\]

Where:

- \((x_i, y_i)\) is the position of router \(i\)
- All positions must lie inside the grid and not inside obstacles

---

## 2. Signal Propagation Model

Use a simplified **log-distance path loss model**:

\[
RSSI*{ij} = P_t - 10\eta \log*{10}(d*{ij} + 1) - \alpha \cdot O*{ij}
\]

Where:

- \(P_t\): transmit power (e.g., -30 dBm)
- \(\eta\): path loss exponent (e.g., 2–3)
- \(d\_{ij}\): Euclidean distance between router \(i\) and cell \(j\)
- \(O\_{ij}\): number of obstacles between router \(i\) and cell \(j\)
- \(\alpha\): obstacle attenuation factor

For each grid cell \(j\):

\[
RSSI*j = \max_i RSSI*{ij}
\]

---

## 3. Objective Functions (Minimization)

Let:

- \(C\) = set of client cells
- \(G\) = set of all grid cells
- \(RSSI\_{min}\) = minimum usable signal

---

### Objective 1: Maximize Coverage

Coverage is defined as the fraction of clients receiving usable signal.

\[
f*1(\mathbf{x}) = -\frac{1}{|C|} \sum*{j \in C} \mathbb{1}(RSSI*j \ge RSSI*{min})
\]

---

### Objective 2: Maximize Mean RSSI at Clients

\[
f*2(\mathbf{x}) = -\frac{1}{|C|} \sum*{j \in C} RSSI_j
\]

---

### Objective 3: Minimize Overlap

Overlap measures redundant coverage by multiple routers:

\[
Overlap*j = \sum_i \mathbb{1}(RSSI*{ij} \ge RSSI\_{min}) - 1
\]

\[
f*3(\mathbf{x}) = \sum*{j \in G} \max(0, Overlap_j)
\]

---

### Final Objective Vector

\[
\mathbf{f}(\mathbf{x}) = [f_1(\mathbf{x}), f_2(\mathbf{x}), f_3(\mathbf{x})]
\]

---

## 4. MOEA/D Framework

### 4.1 Weight Vectors

Generate \(K\) uniformly distributed weight vectors:

\[
\lambda^k = (\lambda_1, \lambda_2, \lambda_3), \quad \sum \lambda_i = 1
\]

Use simplex-lattice or uniform random sampling.

---

### 4.2 Scalarization (Tchebycheff)

For each subproblem \(k\):

\[
g(\mathbf{x} \mid \lambda^k, \mathbf{z}^_) =
\max_i \left( \lambda_i^k \cdot |f_i(\mathbf{x}) - z_i^_| \right)
\]

Where:

- \(\mathbf{z}^_\) is the ideal point:
  \[
  z_i^_ = \min\_{\mathbf{x}} f_i(\mathbf{x})
  \]

---

### 4.3 Neighborhood Structure

For each weight vector:

- Compute Euclidean distances to other weight vectors
- Keep `T` nearest neighbors
- Evolution occurs **within neighborhoods**

---

## 5. Evolutionary Operators

### 5.1 Initialization

- Randomly generate one solution per weight vector
- Router positions sampled uniformly from free grid cells

---

### 5.2 Selection

- For each subproblem \(k\):
  - Select two parents from its neighborhood

---

### 5.3 Crossover (Router-wise)

- Uniform crossover at router level
- Example:
  - Router positions inherited independently from parents

---

### 5.4 Mutation

- With probability \(p_m\):
  - Move a router by ±1 grid cell (valid only)
  - Or re-sample router position randomly

---

### 5.5 Replacement Rule

For offspring \(\mathbf{y}\):

- Evaluate \(\mathbf{f}(\mathbf{y})\)
- Update ideal point \(\mathbf{z}^\*\)
- For each neighbor \(j\):
  - If \(g(\mathbf{y} \mid \lambda^j) \le g(\mathbf{x}\_j \mid \lambda^j)\), replace

---

## 6. Algorithm Loop
