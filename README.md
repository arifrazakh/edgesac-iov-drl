<h1 align="center">EdgeSAC: Graph Neural Soft Actor-Critic for Hierarchical IoV Resource Management</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-6A1B9A" alt="PyG">
  <img src="https://img.shields.io/badge/Gymnasium-RL%20Environment-0081A5" alt="Gymnasium">
  <img src="https://img.shields.io/badge/Domain-5G%20NR%20IoV-5468FF" alt="5G NR IoV">
</p>

<p align="center">
  Repository for the paper <strong>“EdgeSAC: Graph Neural Soft Actor-Critic for Hierarchical IoV Resource Management.”</strong>
</p>

EdgeSAC studies **joint transmit-power and channel-reservation control** for a hierarchical **5G NR Internet of Vehicles (IoV)** network with **macro/micro tiers**, **dual connectivity**, **graph-aware reinforcement learning**, and **NR/MIMO-consistent link modeling**.

The core idea is to represent the radio network as a **graph of base stations**, let a **graph-aware Soft Actor-Critic (SAC)** agent output **continuous per-site control actions**, and then map those actions into feasible radio decisions using an **on-demand scheduler** and **water-filling power allocation**. This keeps the control space continuous while still respecting practical constraints such as finite channel pools, tier budgets, coverage, and the rule that each user can hold **at most one macro and one micro link per slot**.

---

## Overview

Hierarchical IoV resource management is difficult because several objectives are tightly coupled:

- **throughput** must remain high under dense interference,
- **power consumption** must remain controlled,
- **latency** must stay low under mobility,
- **fairness and satisfaction** must remain stable as load changes,
- and all decisions must respect **finite radio resources** and **dual-connectivity limits**.

EdgeSAC addresses this by combining:

1. **Graph-aware interference reasoning** across macro and micro base stations,
2. **Continuous-action SAC** for per-site control,
3. **Scheduler-aligned reservation and admission**,
4. **Shannon-with-gap + rank-adaptive MIMO** for NR-consistent evaluation,
5. and **realistic urban mobility and propagation modeling** for reproducible experiments.

Compared with non-graph or non-scheduler-aware baselines, the method is designed to better align learning with what the network actually does at run time.

---

## What Makes EdgeSAC Different

- **Graph-aware SAC** instead of treating each base station independently
- **Continuous per-base-station control** instead of coarse action discretization
- **Dual-tier scheduling** with at most one macro and one micro link per user
- **On-demand channel activation** instead of blanket reservation
- **Water-filling power split** across active channels
- **3GPP-style UMa/UMi propagation** with Manhattan mobility
- **NR-consistent Shannon-with-gap rate mapping** with **rank-adaptive MIMO**
- **Strong experiment logging** for reward, QoS, latency, fairness, handover, energy, and utilization

---

## Visual Overview

### 1) Hierarchical IoV system model

This figure shows the two-tier deployment used in the paper. Macro cells provide wide-area service, micro cells densify capacity, and vehicles may simultaneously keep **one macro link and one micro link**. The controller manages power and spectrum while mobility triggers handoffs.

![System Model](figs/system-model.png)

### 2) EdgeSAC learning and control pipeline

This figure shows the full control loop. The environment produces the current state, reward, and next state; the graph-aware SAC agent outputs per-site actions; replay memory stores transitions; and actor/critic/value networks are updated from sampled experience.

![EdgeSAC Pipeline](figs/edgesac-pipeline.png)

### 3) Simulation environment rendering

This rendering is useful for visually checking deployment geometry, macro/micro placement, user movement, coverage regions, and qualitative scheduling behavior during training or testing.

![Simulation Setup](figs/sim-setup.png)

---

## Unified System View

```text
Vehicular mobility + traffic demand
              |
              v
Hierarchical IoV environment
(macro tier + micro tier + carrier pools + mobility + PHY)
              |
              v
Base-station graph construction
(node features + edge features)
              |
              v
EdgeSAC agent
(ENGNN actor + graph critics + value network)
              |
              v
Continuous per-BS actions
[power fraction, reservation fraction]
              |
              v
On-demand scheduler
(coverage-aware association, dual connectivity, channel admission)
              |
              v
Water-filling power allocation + SINR/rate evaluation
              |
              v
Reward, diagnostics, replay memory, network updates
```

This design is important because the agent does **not** directly assign users to channels one by one. Instead, it outputs **continuous control signals per base station**, and the environment converts them into feasible scheduling and power-allocation decisions.

---

## Method Summary

### Action space

For each base station `b`, the policy outputs a **2D continuous action**:

- `action[b, 0]` → **power fraction**
- `action[b, 1]` → **channel / reservation fraction**

These are mapped to:

- transmit power under per-tier budgets,
- reserved channel counts under carrier availability,
- user admission under macro/micro constraints,
- and per-channel power allocations via water-filling.

### State representation

The environment provides a **per-base-station observation tensor**. In the simulation code, the observation has shape:

```python
(num_base_stations, 13)
```

The agent then augments this with type information (`macro` vs `micro`) and builds a **static k-nearest-neighbor graph** over base-station locations.

### Scheduler rule

A user can receive:

- **at most one macro link**, and
- **at most one micro link**

in a given time slot.

This keeps the control problem practical and aligns the learning interface with the actual scheduler.

### Reward objective

The environment uses a utility-versus-power tradeoff:

```python
reward = 10.0 * util - 2.0 * power_norm
```

where `util` measures served demand relative to requested demand, and `power_norm` is normalized total transmit power.

---

## Architecture Details

The paper and code together imply the following architecture.

### 1) Environment-side modeling

The environment models:

- macro and micro base stations,
- users with mobility,
- carrier-specific channel pools,
- coverage updates from link-budget calculations,
- load-aware association,
- dual connectivity constraints,
- water-filling power allocation,
- SINR/rate/latency evaluation,
- and reward/diagnostic generation.

### 2) Graph construction

The agent builds a graph where:

- **nodes** are base stations,
- **edges** connect nearby interacting sites,
- **node features** summarize power, utilization, demand, mobility, and interference-related information,
- **edge features** summarize distance, carrier match, tier pairing, and interference context.

In the simulation code, the graph is formed using a **static k-nearest-neighbor graph** with default `n_neighbors = 6`.

### 3) Graph-aware actor and critics

The graph encoder uses edge-aware message passing. In the provided simulation implementation:

- the actor is an **ENGNN-based policy**,
- the critics are **graph-aware Q-networks**,
- the value network is also graph-based,
- and the actor/critics use **two graph blocks** by default.

This gives the policy a structured way to reason about **interference coupling** and **neighbor interactions** instead of making isolated decisions per cell.

### 4) PHY-consistent evaluation

The environment uses:

- **3GPP-style UMa / UMi path loss**,
- **Shannon-with-gap rate mapping**,
- **256-QAM spectral-efficiency cap**,
- **NR overhead factor**,
- **rank-adaptive MIMO**,
- and **water-filling** over active channels.

This makes the benchmark far more realistic than a toy abstract RL environment.

---

## Codebase Tour

This section explains the repository in a more practical way: **what the main objects are**, **what they store**, and **how they interact during one environment step**. For new readers, this is usually easier to understand than looking only at file names.

### Core simulation objects

#### `BaseStation`
A **base station** is the main controllable network entity in the environment.

In the simulation logic, each base station stores and updates information such as:

- its **ID**, **location**, **height**, and **tier type** (`Ma` for macro, `Mi` for micro),
- its current **transmit power**,
- its currently **reserved/assigned channels**,
- its current **coverage radius**,
- and the **per-channel power split** after water-filling.

In simple words, the agent does **not** directly control users. It controls the **base stations**, and each base station then decides how much power and how many channels it will make available in that slot.

A base station is also responsible for several important physical-layer computations, such as:

- computing **3GPP-style path loss**,
- updating its **coverage area** from the current power and carrier settings,
- reserving channels from the available pool,
- and exposing free channels for user admission.

So, when reading the code, think of `BaseStation` as the object that represents **radio infrastructure plus local resource state**.

---

#### `Channel`
A **channel** represents one radio resource unit that belongs to a base station for a given carrier.

Each channel keeps:

- a unique **channel ID**,
- a **carrier frequency**,
- a **bandwidth**,
- a **noise figure**,
- the connected **users** on that channel,
- and the serving **base station**.

Its job is simple but important: it gives the environment a concrete object for bandwidth, thermal noise, carrier separation, and per-user assignment. Instead of saying “the user is connected somehow,” the code says “the user is connected through this exact channel on this exact base station.”

This is important because later the environment uses the channel to compute:

- **noise power**,
- **SINR**,
- **data rate**,
- and whether the channel is already occupied or still free.

---

#### `User`
A **user** represents a vehicle or mobile IoV node moving through the network.

Each user stores:

- its **ID** and **location**,
- its **velocity** and mobility state,
- its current **traffic demand**,
- the channels currently assigned to it,
- per-link or per-channel **SINR values**,
- and the resulting **data rate** and **latency**.

In the code, a user is not just a point on the map. It is a dynamic object that keeps changing over time:

- it moves along the road grid,
- its demand may change,
- it may be connected to a macro BS, a micro BS, or both,
- and its performance is recomputed every step.

So when you read `User`, think of it as the object that carries **mobility**, **service demand**, and **experienced network quality**.

---

#### `MobileNetwork`
`MobileNetwork` is the **main environment class**. It combines everything together.

This is the class that contains:

- the list of **base stations**,
- the list of **users**,
- the pools of **macro and micro channels**,
- the mobility model,
- the observation and action spaces,
- the scheduler,
- the reward function,
- and the metrics reported in each step.

This is the object the RL agent actually interacts with. When the agent calls `step(action)`, it is really calling the `MobileNetwork` environment to:

1. decode the action,
2. update base-station power and channel reservation,
3. run user association,
4. allocate power across active channels,
5. compute SINR, rate, latency, fairness, and other diagnostics,
6. return the next observation, reward, and info dictionary.

If you want to understand the full system behavior, this is the most important class to read first.

---

## How these objects work together

A useful mental model is:

```text
BaseStation -> owns channels and coverage
Channel     -> carries radio resources for one carrier/bandwidth
User        -> moves, requests service, and experiences SINR/rate/latency
MobileNetwork -> orchestrates all of the above inside the RL loop
```

During one slot:

- the **agent outputs actions for base stations**,
- each **base station reserves channels and updates coverage**,
- users search for the **best macro and micro opportunities**,
- channels become the concrete resources used for service,
- and the environment measures the resulting network quality.

That means the code is fundamentally **base-station-centric**, while users and channels are the entities that make the consequences of those decisions visible.

---

## Important environment logic in plain language

### 1) Macro and micro tiers
The environment has two types of base stations:

- **macro BSs** for broad coverage,
- **micro BSs** for denser capacity.

The code assigns different defaults to them, including power budget, height, carrier frequency, coverage behavior, and MIMO rank. This is why the macro tier behaves differently from the micro tier.

### 2) Channel pools
The environment creates separate pools for:

- **macro channels**, and
- **micro channels**.

Each pool is tied to carrier frequencies and per-channel bandwidths. When a base station reserves channels, it does not create them from nowhere; it selects them from its carrier-specific pool.

### 3) Coverage and association
Each base station updates a coverage radius from its power and path-loss model. Users are then checked against that coverage. A user is only eligible for a base station if it is inside that base station’s current service region.

### 4) Dual connectivity
The scheduler follows a practical rule:

- **at most one macro link**, and
- **at most one micro link**

per user per step.

This is one of the most important design choices in the code because it keeps the scheduling feasible and aligned with the problem formulation.

### 5) Water-filling
After users are admitted, the base station does not split power equally in a naive way. Instead, the environment applies **water-filling** over active channels, which gives more useful channels more power under the site budget.

### 6) Mobility
Users move over time, and their motion changes:

- distance to base stations,
- visibility and coverage membership,
- handover behavior,
- interference conditions,
- and ultimately rate and latency.

This is why the environment is dynamic and not just a fixed snapshot optimizer.

---

## Main files and what they do

### `scripts/train_compare.py`
**Main experiment entry point.**

This script is the practical starting point for most users. It usually handles the full experiment workflow:

- training the proposed method,
- running baseline DRL agents,
- evaluating non-learning heuristics,
- collecting metrics,
- and writing logs/results for later plotting.

If you only open one file first, open this one.

---

### `src/iov_power_channel/envs/mobile_network_env.py`
**Main environment logic.**

This file contains the environment-side logic that turns RL outputs into actual network behavior. It is the place to read if you want to understand how:

- base stations are modeled,
- users move,
- channels are reserved,
- users are associated,
- SINR and rates are computed,
- and rewards/diagnostics are generated.

This is the best file for understanding the **system model in code form**.

---

### `src/iov_power_channel/agents/engnn_sac.py`
**Proposed EdgeSAC / ENGNN-SAC implementation.**

This file contains the graph-aware SAC agent. It is responsible for:

- building or consuming the **base-station graph**,
- running the **graph encoder**,
- producing **continuous per-BS actions**,
- maintaining the actor/critic/value networks,
- and performing SAC updates from replay samples.

This is the best file for understanding the **learning method itself**.

---

### `src/iov_power_channel/baselines/sb3_agents.py`
**Baseline RL agents.**

This module contains standard baseline agents such as SAC, PPO, and TD3. These are useful for showing whether the gain comes from:

- using RL in general, or
- using the proposed **graph-aware structured control**.

---

### `src/iov_power_channel/baselines/heuristics.py`
**Non-learning baselines.**

This file contains hand-designed baseline methods, such as fixed-power or rule-based association/control strategies. These baselines matter because they show how much improvement learning provides over simpler engineering rules.

---

### `src/iov_power_channel/utils/common.py`
**Shared utilities.**

This module is for helper logic reused across training, evaluation, logging, or plotting.

---

### `results/`
**Saved experiment outputs.**

This folder stores CSV logs and evaluation results. These files are useful for:

- reproducing plots,
- checking convergence,
- comparing algorithms,
- and preparing tables/figures for a paper.

---

### `graphs/`
**Prepared result figures.**

This folder contains the ready-made performance plots used for visual comparison.

---

### `figs/`
**Explanatory figures for the paper and README.**

This folder contains the system model, the learning/control pipeline, and the simulation-rendering figures used to explain the project visually.

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

## Environment and Modeling Details

The environment is designed to be more realistic than a generic benchmark. Important modeling choices include:

- **3GPP-style UMa / UMi path loss**
- **Manhattan-grid mobility**
- **Shannon-with-gap link abstraction**
- **strict 256-QAM spectral-efficiency cap**
- **rank-adaptive MIMO**
- **load-aware association**
- **water-filling power split across active channels**
- **per-step tracking of QoS, latency, fairness, satisfaction, overlap, outage, and handover metrics**

From the simulation implementation, the environment uses the following representative settings:

```python
NR_OVERHEAD = 0.85
SNR_GAP_DB = 1.5
MIMO_MAX_RANK = {'Ma': 4, 'Mi': 8}
observation_shape = (num_base_stations, 13)
action_shape = (num_base_stations, 2)
```

These details are important because they define the actual control problem seen by the agent.

---

## What the Code Is Doing Internally

A useful way to understand the code is to follow one environment step.

### Step-by-step logic

1. **The agent observes the network state** for all base stations.
2. **A graph is built** over the base stations.
3. **The actor outputs per-BS actions** for power and reservation fractions.
4. **Actions are clamped/scaled** to valid ranges.
5. **Each base station updates power and tentative channel reservation.**
6. **Coverage is refreshed** based on current power and carrier settings.
7. **The scheduler assigns users** with the rule “at most one macro and one micro link.”
8. **Water-filling allocates power** across active channels.
9. **SINR, data rate, latency, and satisfaction** are computed.
10. **Reward and diagnostic metrics** are recorded.
11. **The transition is pushed to replay memory.**
12. **SAC updates** improve the actor, critics, and temperature parameter.

That is the practical control loop implemented by the project.

---

## Metrics Logged During Training and Evaluation

The repository is useful not only for training, but also for analysis.

### Common logged quantities

The provided code and result files track metrics such as:

- **total data rate**
- **macro and micro data rate**
- **average rate**
- **total, macro, and micro power**
- **energy efficiency**
- **latency** and percentile latency
- **number of associated users**
- **channel utilization**
- **QoS mean**
- **Jain fairness**
- **handover counts and handover rate**
- **SINR and rate percentiles**
- **outage ratios**
- **coverage statistics**
- **overlap ratio**
- **reward decomposition**

### Why this matters

This logging makes the repository well suited for:

- algorithm comparison,
- reviewer-requested analysis,
- fairness and robustness studies,
- scalability studies across user loads,
- and paper-quality plot generation.

---

## Included Result Figures

The repository already contains several figures under `graphs/`.

| Reward | QoS / Latency |
|---|---|
| ![Reward](graphs/reward.png) | ![QoS](graphs/qos.png) |

| Satisfaction / Service | Energy / Efficiency |
|---|---|
| ![Satisfaction](graphs/satisfy2.png) | ![Energy](graphs/energy.png) |

---

## Installation

### Standard installation

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
pip install -e .
```

### Notes

This project relies on PyTorch, Gymnasium, and graph-learning dependencies. If your setup uses PyTorch Geometric, make sure your **PyTorch** and **PyG** versions are compatible.

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

## How to Read the Repository for the First Time

A good reading order is:

1. `README.md`
2. `figs/system-model.png`
3. `figs/edgesac-pipeline.png`
4. `figs/sim-setup.png`
5. `scripts/train_compare.py`
6. `src/iov_power_channel/envs/mobile_network_env.py`
7. `src/iov_power_channel/agents/engnn_sac.py`
8. `src/iov_power_channel/baselines/`
9. `results/` and `graphs/`

This order moves from the high-level idea to the implementation details.

---

## Reproducibility Notes

To keep experiments reproducible, the project is designed around:

- a fixed environment structure,
- a graph-based state representation,
- explicit logging to CSV,
- saved comparison outputs,
- and ready-made figures under `graphs/`.

For robust reproduction, keep the following consistent across runs:

- number and placement of base stations,
- number of users,
- area size and mobility parameters,
- PyTorch / PyG version compatibility,
- training steps and evaluation episodes,
- and output file naming.

---

## Recommended README Usage for This Repository

This README is written to serve **three audiences at once**:

- **new readers**, who want to understand the idea quickly,
- **reviewers or collaborators**, who want to verify the method and experiment logic,
- and **future you**, who may return later and need a fast reminder of how the code is organized.

That is why the document includes:

- a plain-language overview,
- a clear code walkthrough,
- visual figures,
- run commands,
- and implementation-level detail.

---

## Contact

For questions about the paper or repository, please open an issue in the GitHub project or contact the authors listed in the manuscript.
