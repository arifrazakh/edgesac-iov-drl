# IoV Power–Channel Control

GitHub-ready research codebase for **joint transmit-power and channel-allocation control** in a 5G/NR IoV cellular environment.

This repository cleans the duplicated notebook code into **one environment** and supports:

- **Proposed method:** `ENGNNSAC` (edge-aware graph SAC)
- **DRL baselines:** `PPO`, `A2C`
- **Non-DRL baselines:** `Random`, `LoadAwareHeuristic`, `MaxPowerMaxChannel`

## What is included

- One unified environment where each base station action is:

  - `action[i, 0]` → **power fraction**
  - `action[i, 1]` → **channel fraction**

- Joint reservation of channels **from the action itself**
- On-demand user scheduling on the reserved channels
- Water-filling power split across active channels
- Metrics and comparison runner for the proposed method, DRL baselines, and non-DRL baselines
- GitHub-ready project layout

## Repository layout

```text
iov_power_channel_repo/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── iov_power_channel/
│       ├── envs/
│       │   └── mobile_network_env.py
│       ├── agents/
│       │   └── engnn_sac.py
│       ├── baselines/
│       │   ├── heuristics.py
│       │   └── sb3_agents.py
│       └── utils/
│           └── common.py
├── scripts/
│   └── train_compare.py
└── results/
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
pip install -e .
```

## Run

Train and compare everything:

```bash
python scripts/train_compare.py --mode all --train-steps 30000 --eval-episodes 5
```

Only the proposed method:

```bash
python scripts/train_compare.py --mode proposed --train-steps 30000
```

Only DRL baselines:

```bash
python scripts/train_compare.py --mode baselines --train-steps 30000
```

Only non-DRL baselines:

```bash
python scripts/train_compare.py --mode heuristics --eval-episodes 10
```

## Main outputs

Saved under `results/`:

- `comparison_summary.csv`
- `per_episode_metrics.csv`
- `comparison_bar.png`

## GitHub steps

```bash
git init
git add .
git commit -m "Initial commit: IoV power-channel control repo"

# create empty repo on GitHub first, then:
git remote add origin https://github.com/<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```

## Notes

- This version is built from the code you pasted in chat and **does not yet include your paper PDF or extra external files**, because they were not attached here.
- Once you upload the paper and any additional baseline code, this repo can be extended to match the manuscript wording more tightly.
