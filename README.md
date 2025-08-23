# social-curiosity

This project explores **computational social intrinsic motivation (SIM)** in a simple multi-agent reinforcement learning setting. Two agents in a **GridWorld** must cooperate to open a door and collect coins.

We compare a **baseline (extrinsic rewards only)** against an agent augmented with a **social curiosity reward** — a bonus when an agent's proximity enables its teammate to explore new states. This improves coordination and sample efficiency.

---

## ✨ Highlights
- **Custom 5×5 GridWorld** with two agents, pressure plates, door, and coins.
- **Social curiosity intrinsic reward**: rewards agents when their teammate explores a new state nearby.
- **Dual implementation approach**:
  - **Tabular Q-learning** for clarity and speed (CPU only)
  - **Deep RL (PPO)** with PettingZoo for modern baseline
- **Easy comparison**: baseline vs SIM performance curves for both implementations

---

## 🏗️ Project Structure
```
social-curiosity/
├── src/                    # Shared utilities and common code
├── tabular/               # Tabular Q-learning implementation
│   ├── src/               # Tabular-specific code
│   └── config/            # Tabular configuration
├── deep/                  # Deep RL implementation
│   ├── src/               # Deep-specific code
│   └── config/            # Deep configuration
├── results/               # Experiment results
├── plots/                 # Generated plots
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

---

## 🚀 Quickstart

### Installation
```bash
# Python 3.10+ required
python -m venv .venv && source .venv/bin/activate

# Install with uv (recommended)
uv sync

# Or with pip
pip install .
```

### Tabular Implementation
```bash
# Baseline (extrinsic only)
python tabular/src/train.py --run_name baseline --intrinsic_coef 0.0

# With social curiosity intrinsic motivation
python tabular/src/train.py --run_name sim --intrinsic_coef 0.2

# Compare curves
python tabular/src/plot_runs.py
```

### Deep Learning Implementation
```bash
# Baseline (extrinsic only)
python deep/src/train.py --run_name baseline --intrinsic_coef 0.0

# With social curiosity intrinsic motivation
python deep/src/train.py --run_name sim --intrinsic_coef 0.2

# Compare curves
python deep/src/plot_runs.py
```

---

## 📦 Dependencies

Core dependencies include:
- **numpy**, **matplotlib** - Core numerical and plotting
- **gymnasium**, **pettingzoo** - Multi-agent environments
- **stable-baselines3**, **torch** - Deep reinforcement learning
- **wandb**, **tensorboard** - Experiment tracking

See [`pyproject.toml`](pyproject.toml) for complete dependency list.

---

## 📊 Results & Plots

Experiment results are saved in `results/` directory, organized by implementation type:
- `results/tabular/` - Tabular Q-learning results
- `results/deep/` - Deep RL results

Generated plots are saved in `plots/` directory with the same structure.

---

## 🔧 Development

Install development dependencies:
```bash
uv sync --group dev
```

Run tests:
```bash
pytest
```

Format code:
```bash
black .
isort .
ruff check --fix
```
