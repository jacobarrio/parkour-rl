# Parkour RL

Just a simple reinforcement learning environment where an agent learns to jump across gaps. I built this to get a better handle on PPO implementation details. It uses Pymunk for physics and Pygame for rendering.

## Features
- **Physics-based**: Uses Pymunk for 2D rigid body physics (gravity, friction, impulses).
- **PPO from scratch**: A clean, minimal implementation of Proximal Policy Optimization in PyTorch.
- **Curriculum Learning**: The environment gets harder as you level up (wider gaps).

## Installation

You'll need a few standard libraries:

```bash
pip install torch pygame pymunk gymnasium numpy
```

## Quick Start

To train the agent from scratch:
```bash
python train.py
```
This will save checkpoints to `parkour_agent.pt`.

To watch the trained agent play:
```bash
python render.py
```
Controls: `ESC` to quit, `R` to reset.

## Environment Specs

### Observation Space (9D)
The agent sees a normalized vector containing:
- Position (x, y)
- Velocity (vx, vy)
- Angle & Angular Velocity
- Grounded state (0 or 1)
- Distance to nearest edge
- Distance to next platform

### Action Space
Discrete(4):
0. Do nothing
1. Move Left (force)
2. Move Right (force)
3. Jump (impulse, only if grounded)

### Rewards
- **Progress**: +2.0 * distance moved (only for new territory)
- **Survival**: +0.1 per frame
- **Velocity**: Bonus for moving right
- **Penalties**: -10 for falling off
- **Success**: +100 for reaching the end

## Curriculum Levels
1. **Lvl 1**: Tiny gaps (5-15px). Basically walking.
2. **Lvl 2**: Small jumps (15-40px).
3. **Lvl 3**: Real jumps (40-80px).
4. **Lvl 4**: Hard jumps (80-120px).
5. **Lvl 5**: Extreme (100-150px). Good luck.

## Results
On difficulty 1, the agent usually hits ~3500 reward after a while. It figures out that running right is good, but takes a bit longer to learn precise jumping.

## Structure
- `environment.py`: The gym environment and physics logic.
- `train.py`: The PPO training loop.
- `render.py`: Script to load a model and watch it run.

---
*Note: This is just a weekend project, so the code might be a bit rough in places, but it gets the job done.*
