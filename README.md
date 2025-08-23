# social-curiosity

This project explores **computational social intrinsic motivation (SIM)** in a simple multi-agent reinforcement learning setting. Two agents in a **GridWorld** must cooperate to open a door and collect coins.  

We compare a **baseline (extrinsic rewards only)** against an agent augmented with a **social curiosity reward** — a bonus when an agent’s proximity enables its teammate to explore new states. This improves coordination and sample efficiency.  

---

## ✨ Highlights
- **Custom 5×5 GridWorld** with two agents, pressure plates, door, and coins.  
- **Social curiosity intrinsic reward**: rewards agents when their teammate explores a new state nearby.  
- **Independent tabular Q-learning** for clarity and speed (CPU only).  
- **Optional deep RL (PPO)** extension for a modern baseline.  
- **Easy plots**: baseline vs SIM performance curves.  

---

## 🚀 Quickstart
```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Baseline (extrinsic only)
python src/train.py --run_name baseline --intrinsic_coef 0.0

# With social curiosity intrinsic motivation
python src/train.py --run_name sim --intrinsic_coef 0.2

# Compare curves
python src/plot_runs.py
