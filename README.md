# EdgeSAC: Graph Neural Soft Actor-Critic for Hierarchical IoV Resource Management

Repository for the paper **"EdgeSAC: Graph Neural Soft Actor-Critic for Hierarchical IoV Resource Management"**.

This codebase studies **joint transmit-power and channel-reservation control** for a hierarchical 5G NR Internet of Vehicles (IoV) network with **macro/micro tiers**, **dual connectivity**, **graph-aware reinforcement learning**, and **NR/MIMO-consistent link modeling**.

---

## Overview

EdgeSAC is a graph-aware Soft Actor-Critic framework executed at the edge for **continuous per-base-station radio control**. The controller outputs per-site actions for:

- **power fraction**
- **channel / reservation fraction**

The environment then performs:

- **coverage-aware and load-aware association**
- **at most one macro and one micro link per user**
- **on-demand fixed-size channel activation**
- **water-filling over active channels**
- **Shannon-with-gap rate mapping with rank-adaptive MIMO**

The repository is organized for reproducible comparisons between the proposed graph-aware controller and baseline DRL / heuristic policies.

---

## Main Highlights

- Graph-aware SAC for interference-coupled hierarchical IoV resource control
- 3GPP-style UMa/UMi propagation with Manhattan mobility
- Continuous control without action discretization
- Dual-tier scheduling with macro + micro connectivity constraints
- Rank-adaptive MIMO and Shannon-with-gap spectral-efficiency modeling
- Comparison against DRL baselines and non-learning heuristics
- Ready-to-use result CSVs and training/evaluation figures

---

## System Model

> GitHub does not render PDF figures inline in README files. For that reason, PNG versions are recommended for the README even if the original paper figures are stored as PDF.

### Hierarchical IoV system model

![System Model](figs/system-model.png)

### EdgeSAC pipeline

![EdgeSAC Pipeline](figs/edgesac-pipeline.png)

---

## Selected Result Figures

These plots are already present in the repository under `graphs/` and can be rendered directly by GitHub.

| Reward | QoS / Latency |
|---|---|
| ![Reward](graphs/reward.png) | ![QoS](graphs/qos.png) |

| Satisfaction / Handover | Energy / Fairness |
|---|---|
| ![Satisfaction](graphs/satisfy2.png) | ![Energy](graphs/energy.png) |

---

## Repository Structure

```text
.
├── figs/
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

## Method at a Glance

### Proposed controller

- **EdgeSAC / ENGNNSAC** (`src/iov_power_channel/agents/engnn_sac.py`)
- Edge-update graph network with SAC-style continuous control
- Actor and critics operate on the base-station graph rather than independent per-cell features

### Environment

- **Mobile network environment** (`src/iov_power_channel/envs/mobile_network_env.py`)
- 5G NR-like macro/micro deployment
- 3GPP-style path loss
- Manhattan-grid mobility
- Coverage refresh from planning power / carrier assumptions
- On-demand reservation and user scheduling
- Water-filling power split across active channels

### Baselines

- **DRL baselines** in `src/iov_power_channel/baselines/sb3_agents.py`
  - SAC
  - PPO
  - TD3
- **Heuristic baselines** in `src/iov_power_channel/baselines/heuristics.py`
  - fixed / reservation-based baselines
  - max-SINR style baseline
  - demand-aware heuristic variants

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

## What the Action Represents

For each base station `i`, the policy outputs a 2D continuous action:

- `action[i, 0]` -> **power fraction**
- `action[i, 1]` -> **channel / reservation fraction**

These actions are mapped to:

- site transmit power under tier budgets
- integer reserved-channel counts
- scheduler-side admission decisions
- per-channel power allocation through water-filling

This keeps the controller fully continuous while the environment enforces practical capacity and dual-connectivity constraints.

---

## Key Modeling Assumptions

- Hierarchical deployment with **4 macro** and **32 micro** base stations
- **100-user** default scenario, with scalability experiments for **200** and **300** users
- Manhattan mobility with road spacing and intersection behavior
- Macro carrier range around **3.4-3.8 GHz**
- Micro carrier range around **24.5-28.5 GHz**
- 3GPP-style UMa / UMi path loss
- Shannon-with-gap mapping with **256-QAM cap** and **rank-adaptive MIMO**
- Per-user service with at most **one macro** and **one micro** link per scheduling slot

---

## Results Snapshot

### Scalability across user-load regimes

| Users | Policy | Reward | Total Rate (Mbps) | EE (Mb/J) | Latency (ms) | Satisfied Users |
|---:|---|---:|---:|---:|---:|---:|
| 100 | EdgeSAC | 3.75 | 7601 | 33.21 | 20.0 | 40 |
| 200 | EdgeSAC | 3.33 | 14086 | 59.12 | 27.1 | 71 |
| 300 | EdgeSAC | 3.13 | 18961 | 85.02 | 31.4 | 99 |

### Aggregate comparison with learning and non-learning baselines

| Policy | Total Rate (Mbps) | Reserved BW (MHz) | SEavg (b/s/Hz) | EE (Mb/J) | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| EdgeSAC | 7601 | 953 | 9.38 | 33.21 | 20 |
| SAC | 7821 | 934 | 9.85 | 29.88 | 28 |
| PPO | 4614 | 737 | 7.37 | 30.88 | 48 |
| TD3 | 8216 | 961 | 10.06 | 20.79 | 19 |
| FPFR | 6696 | 3024 | 2.61 | 27.90 | 44 |
| AOR-BP | 7931 | 6048 | 1.54 | 16.52 | 44 |
| MS-FR | 6704 | 6048 | 1.30 | 27.94 | 49 |

**Takeaway:** EdgeSAC maintains strong spectral utilization while achieving the best energy-efficiency / latency balance among the compared methods.

---

## Result Files

Example outputs stored in `results/` include:

- `training_log.csv`
- `edgesac_hold_eval.csv`
- `edgesac_users200.csv`
- `edgesac_users300.csv`
- `sac_mlp_log_200.csv`, `sac_mlp_log_300.csv`
- `ppo_mlp_log_200.csv`, `ppo_mlp_log_300.csv`
- `td3_mlp_log200.csv`, `td3_mlp_log_300.csv`
- `fix_power_reser_eval.csv`
- `ma_power_demand_eval.csv`
- `max_sinr_eval.csv`

These files can be used to regenerate the paper figures, compare training trends, and analyze scalability under different user-load regimes.

---

## Reproducibility Notes

- Use the same train/eval seed configuration across all policies when reproducing figures.
- Keep the environment parameters fixed when comparing learning and heuristic baselines.
- For paper-style plots, use moving averages or last-ten-episode averages consistently.
- Prefer PNG assets for README display, even if paper figures are preserved in PDF for publication quality.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{raza2026edgesac,
  title   = {EdgeSAC: Graph Neural Soft Actor-Critic for Hierarchical IoV Resource Management},
  author  = {Raza, Arif and Borhan, Uddin Md. and Che, Yueling and Jie, Chen and Wang, Lu},
  journal = {IEEE Transactions on Mobile Computing},
  year    = {2026}
}
```

---

## Contact

For research collaboration or questions about the implementation, please open an issue in this repository or contact the paper authors.
