# WiFi Coverage Simulation - Step A

A minimal working prototype for simulating wireless coverage with static access points and mobile router agents.

## Quick Start

```powershell
cd wifi_simulation_1
.\venv\Scripts\activate
python simulation.py
```

## Physics Model

### Log-Distance Path-Loss Model

```
RSSI(d) = P_tx - PL_0 - 10·n·log10(d/d0) - obstacle_penalty
```

| Parameter        | Default | Description                                |
| ---------------- | ------- | ------------------------------------------ |
| P_tx             | 20 dBm  | Transmit power (typical WiFi AP)           |
| PL_0             | 40 dB   | Path loss at 1m reference distance         |
| n                | 3.0     | Path-loss exponent (indoor with obstacles) |
| d_0              | 1 m     | Reference distance                         |
| Obstacle penalty | 15 dB   | Per-obstacle signal attenuation            |

### Coverage Threshold

A cell is "covered" if RSSI ≥ **-70 dBm** (typical WiFi receiver sensitivity).

## Simulation Parameters

Edit the constants at the top of `simulation.py`:

| Parameter            | Value | Description           |
| -------------------- | ----- | --------------------- |
| `GRID_SIZE`          | 50    | 50×50 meter area      |
| `OBSTACLE_DENSITY`   | 0.17  | ~17% obstacles        |
| `NUM_STATIC_APS`     | 3     | Fixed access points   |
| `NUM_CLIENTS`        | 20    | Static client devices |
| `NUM_MOBILE_ROUTERS` | 4     | Mobile relay agents   |

## Mobile Router Kinematics

Uses the **unicycle model**:

- Max speed: 2 m/s
- Max turn rate: 45°/s

```
x_dot = v · cos(θ)
y_dot = v · sin(θ)
θ_dot = ω
```

## Visualization

The heatmap shows:

- 🔺 Red triangles: Static APs
- 🔵 Blue circles: Covered clients
- ⬜ Gray circles: Uncovered clients
- 🟩 Green squares: Mobile routers
- ⚫ Gray cells: Obstacles
- ⬜ White dashed line: -70 dBm coverage boundary

## Animation

To see mobile routers move with random-walk behavior:

```python
results = run_simulation(animate=True)
```
