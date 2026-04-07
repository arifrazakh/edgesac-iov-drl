# EdgeSAC: Graph Neural Soft Actor-Critic for Hierarchical IoV Resource Management

Repository for the paper **“EdgeSAC: Graph Neural Soft Actor-Critic for Hierarchical IoV Resource Management.”**

EdgeSAC studies **joint transmit-power and channel-reservation control** for a hierarchical **5G NR Internet of Vehicles (IoV)** network with **macro/micro tiers**, **dual connectivity**, **graph-aware reinforcement learning**, and **NR/MIMO-consistent link modeling**.

The main idea is simple: instead of treating each base station independently or discretizing radio decisions, EdgeSAC builds a **graph over base stations**, reasons about **interference coupling**, and outputs **continuous per-site control actions**. A scheduler then turns those actions into feasible radio allocations under coverage, channel-pool, and dual-connectivity constraints.

---

## Highlights

- **Graph-aware SAC** for interference-coupled radio control across macro and micro tiers
- **Continuous per-base-station actions** for power and reservation control
- **Scheduler-aligned design** with at most one macro and one micro link per user per slot
- **3GPP-style UMa/UMi propagation**, Manhattan mobility, and NR-like PHY abstraction
- **Shannon-with-gap + rank-adaptive MIMO** without action discretization
- **Learning and non-learning comparisons** with ready-made logs, results, and figures

---

## Visual Overview

### 1) Hierarchical IoV system model

This figure illustrates the two-tier IoV deployment used throughout the paper: macro cells provide wide-area coverage, micro cells densify capacity, and vehicles may hold **one macro link and one micro link** simultaneously. The controller manages power and spectrum decisions while handoffs occur under mobility.

![System Model](figs/system-model.png)

### 2) EdgeSAC learning pipeline

This figure shows the control loop used by the method. The mobile environment produces the current state, reward, and next state; the graph-aware SAC agent outputs per-site actions; replay memory stores transitions; and actor/critic/value networks are updated from sampled experience.

![EdgeSAC Pipeline](figs/edgesac-pipeline.png)

### 3) Simulation environment rendering

This rendering shows how the environment looks during simulation. It is useful for visually checking deployment geometry, macro/micro placement, user movement, coverage behavior, and qualitative scheduling dynamics during training or testing.

![Simulation Setup](figs/sim-setup.png)

---

## What Problem This Repository Solves

Hierarchical IoV resource management is difficult because the controller must balance several coupled objectives at once:

- high throughput under dense interference,
- low power consumption,
- stable service under mobility,
- fairness and user satisfaction,
- and realistic admission under finite channel pools.

EdgeSAC addresses this by combining a **permutation-equivariant graph encoder**, **continuous-action SAC**, **on-demand reservation/scheduling**, and **NR/MIMO-consistent evaluation**. In the paper, the method is positioned against baselines that either lack graph-aware inter-BS reasoning, lack continuous per-BS radio control, or are not fully aligned with scheduler-aware dual-tier operation.

---

## How EdgeSAC Works

At each time step, every base station outputs a **2D continuous action**:

- `action[i, 0]` → **power fraction**
- `action[i, 1]` → **channel / reservation fraction**

These actions are mapped to:

- site transmit power under per-tier budgets,
- reserved channel counts under carrier availability,
- scheduler-side admission decisions,
- and per-channel power allocations via **water-filling**.

The environment then:

1. updates per-site coverage,
2. reserves channels based on the action,
3. performs coverage-aware and load-aware association,
4. allows at most **one macro and one micro link per user**,
5. computes per-channel power splits,
6. evaluates SINR, rate, latency, fairness, and satisfaction,
7. and returns the next observation and reward.

This keeps the agent’s action space fully continuous while still respecting practical network constraints.

---

## Repository Structure

```text
.
├── figs/
│   ├── edgesac-pipeline.png
│   ├── system-model.png
│   ├── sim-setup.png
│   ├── mimo_framework.pdf
│   ├── mimo_model.pdf
│   └── sim-setup.pdf
├── graphs/
│   ├── energy.png
│   ├── qos.png
│   ├── reward.png
│   └── satisfy2.png
├── pyproject.toml
├── README.md
├── requirements.txt
├── results/
│   ├── edgesac_hold_eval.csv
│   ├── edgesac_users200.csv
│   ├── edgesac_users300.csv
│   ├── fix_power_reser_eval.csv
│   ├── ma_power_demand_eval.csv
│   ├── max_sinr_eval.csv
│   ├── ppo_mlp_log_200.csv
│   ├── ppo_mlp_log_300.csv
│   ├── sac_mlp_log_200.csv
│   ├── sac_mlp_log_300.csv
│   ├── td3_mlp_log200.csv
│   ├── td3_mlp_log_300.csv
│   └── training_log.csv
├── scripts/
│   └── train_compare.py
└── src/
    └── iov_power_channel/
        ├── agents/
        │   └── engnn_sac.py
        ├── baselines/
        │   ├── heuristics.py
        │   └── sb3_agents.py
        ├── envs/
        │   └── mobile_network_env.py
        └── utils/
            └── common.py
```

---

## Code Walkthrough

This section explains what each main file is for, so a new reader can understand the codebase quickly.

### `scripts/train_compare.py`
Main experiment entry point.

This script is the practical starting point for most users. It is intended to:

- train the proposed EdgeSAC policy,
- run baseline DRL agents,
- evaluate non-learning heuristics,
- collect metrics,
- and write logs/results for later plotting and comparison.

If you want to reproduce the main comparisons of the repository, this is the first file to run.

### `src/iov_power_channel/agents/engnn_sac.py`
Proposed **EdgeSAC / ENGNNSAC** implementation.

This module contains the graph-aware Soft Actor-Critic agent. The policy operates on a **base-station graph** rather than on isolated cell features, so the actor and critics can reason about interference coupling, neighboring load, and tier interactions. In the paper, EdgeSAC uses a permutation-equivariant graph representation and continuous control rather than hybrid or discretized action selection.

### `src/iov_power_channel/envs/mobile_network_env.py`
Hierarchical IoV simulation environment.

This is the core environment that turns RL actions into network behavior. Based on the uploaded simulation code and the paper description, the environment models:

- macro and micro base stations,
- vehicular users with mobility,
- carrier-aware channel pools,
- coverage updates from link-budget assumptions,
- on-demand channel reservation,
- dual connectivity constraints,
- water-filling power allocation,
- SINR/rate/latency evaluation,
- and reward/diagnostic generation.

This file is the best place to understand how a continuous action becomes an actual radio decision.

### `src/iov_power_channel/baselines/sb3_agents.py`
Standard DRL baselines.

This module contains baseline learning agents used for comparison with the proposed method, such as **SAC**, **PPO**, and **TD3**. These baselines help show what improvement comes from the graph-aware design rather than from using RL in general.

### `src/iov_power_channel/baselines/heuristics.py`
Non-learning baselines.

This file contains hand-designed baseline strategies such as reservation-based or max-SINR-style rules. These baselines are important because they show how much benefit the learned policy provides over conventional scheduling/control logic.

### `src/iov_power_channel/utils/common.py`
Shared utilities.

This module is intended for reusable helper functions shared across training, evaluation, logging, or plotting workflows.

### `results/`
Saved experiment outputs.

This folder contains the generated CSV files for training and evaluation. These logs can be used to reproduce paper plots, compare policies, and inspect scalability across 100-, 200-, and 300-user scenarios.

### `graphs/`
Prepared result figures.

This folder contains ready-to-show figures for reward, QoS, satisfaction, and energy/fairness trends.

### `figs/`
Paper and README figures.

This folder contains the system-model, pipeline, and simulation-setup figures used to explain the method visually.

---

## Environment and Modeling Details

The uploaded paper and simulation code show that the environment is not a toy abstraction. It includes several important modeling choices that make the benchmark more realistic:

- **3GPP-style UMa / UMi path loss** for macro and micro tiers
- **Manhattan-grid mobility** with road structure and turning behavior
- **Shannon-with-gap rate mapping**
- **256-QAM spectral-efficiency cap**
- **rank-adaptive MIMO**
- **load-aware association**
- **water-filling power split across active channels**
- **per-step metrics** such as latency, QoS, fairness, satisfaction, handovers, and utilization

In the uploaded simulation code, the environment uses:

- `NR_OVERHEAD = 0.85`
- `SNR_GAP_DB = 1.5`
- `MIMO_MAX_RANK = {'Ma': 4, 'Mi': 8}`
- an observation with **13 features per base station**
- and a continuous action space of shape **(num_base_stations, 2)**.

---

## Graph State and Agent View

In the paper, the state is modeled as a graph whose:

- **nodes** represent base stations,
- **edges** represent nearby interacting base-station pairs,
- **node features** include tier flag, power/utilization, demand/mobility summaries, and interference-related information,
- **edge features** include distance, carrier match, type pairing, and power/interference context.

The paper also states that the implementation constructs the inter-BS graph as a **static k-nearest-neighbor graph** over base-station coordinates with **default `k = 6`** and bidirectional message passing. This is a key part of why the method can reason about coupled interference rather than treating each site independently.

---

## Reward and Control Objective

The uploaded simulation code uses a reward shaped around **utility versus normalized power**, specifically:

- reward contribution from service utility,
- penalty from normalized power use,
- plus environment-level diagnostics for fairness, latency, outages, satisfaction, and handover behavior.

In the simulation file, the environment reward is implemented as:

```python
reward = 10.0 * util - 2.0 * power_norm
```

This is consistent with the paper’s description that the controller balances demand satisfaction against normalized power use rather than simply maximizing throughput.

---

## Key Files in `results/`

The repository already includes multiple result logs. For example:

- `training_log.csv` — overall training trace for the proposed method
- `edgesac_hold_eval.csv` — evaluation summary for the proposed method
- `edgesac_users200.csv`, `edgesac_users300.csv` — scalability experiments under heavier user loads
- `sac_mlp_log_200.csv`, `ppo_mlp_log_200.csv`, `td3_mlp_log200.csv` — baseline logs for comparison
- `fix_power_reser_eval.csv`, `ma_power_demand_eval.csv`, `max_sinr_eval.csv` — non-learning baseline evaluations

These make the repository useful not only for training, but also for **plot reproduction**, **post-hoc analysis**, and **paper/report preparation**.

---

## Selected Result Figures

The repository already contains result figures under `graphs/`.

| Reward | QoS / Latency |
|---|---|
| ![Reward](graphs/reward.png) | ![QoS](graphs/qos.png) |

| Satisfaction / Handover | Energy / Fairness |
|---|---|
| ![Satisfaction](graphs/satisfy2.png) | ![Energy](graphs/energy.png) |

---

## Reported Performance Snapshot

From the paper’s reported scalability table, EdgeSAC maintains strong performance across increasing user loads:

| Users | Reward | Total Rate (Mbps) | EE (Mb/J) | Latency (ms) | Satisfied Users |
|---:|---:|---:|---:|---:|---:|
| 100 | 3.75 | 7601  | 33.21 | 20.0 | 40 |
| 200 | 3.33 | 14086 | 59.12 | 27.1 | 71 |
| 300 | 3.13 | 18961 | 85.02 | 31.4 | 99 |

These numbers suggest that the method scales well with heavier traffic while keeping the utility–power tradeoff competitive.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### Run the full comparison suite

```bash
python scripts/train_compare.py --mode all --train-steps 30000 --eval-episodes 5
```

### Train only the proposed method

```bash
python scripts/train_compare.py --mode proposed --train-steps 30000
```

### Train only the DRL baselines

```bash
python scripts/train_compare.py --mode baselines --train-steps 30000
```

### Evaluate heuristic baselines only

```bash
python scripts/train_compare.py --mode heuristics --eval-episodes 10
```

---

## Suggested Reading Order

If you are opening the repository for the first time, this order is a good way to understand it:

1. `README.md`
2. `figs/system-model.png`
3. `figs/edgesac-pipeline.png`
4. `figs/sim-setup.png`
5. `scripts/train_compare.py`
6. `src/iov_power_channel/envs/mobile_network_env.py`
7. `src/iov_power_channel/agents/engnn_sac.py`
8. `src/iov_power_channel/baselines/`
9. `results/` and `graphs/`



---

## Contact

For questions about the paper or the repository, please open an issue in the GitHub project or contact the authors listed in the manuscript.
