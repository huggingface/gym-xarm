# gym-xarm

A gym environment for xArm

<td><img src="http://remicadene.com/assets/gif/simxarm_tdmpc.gif" width="50%" alt="TDMPC policy on xArm env"/></td>


## Installation

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n xarm python=3.10 && conda activate xarm
```

Install gym-xarm:
```bash
pip install gym-xarm
```


## Quickstart

```python
# example.py
import gymnasium as gym
import gym_xarm

env = gym.make("gym_xarm/XarmLift-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

To use this [example](./example.py) with `render_mode="human"`, you should set the environment variable `export MUJOCO_GL=glfw` or simply run
```bash
MUJOCO_GL=glfw python example.py
```

## Description for `Lift` task

The goal of the agent is to lift the block above a height threshold. The agent is an xArm robot arm and the block is a cube.

### Action Space

The action space is continuous and consists of four values [x, y, z, w]:
- [x, y, z] represent the position of the end effector
- [w] represents the gripper control

### Observation Space

Observation space is dependent on the value set to `obs_type`:
- `"state"`: observations contain agent and object state vectors only (no rendering)
- `"pixels"`: observations contains rendered image only (no state vectors)
- `"pixels_agent_pos"`: contains rendered image and agent state vector


## Contribute

Instead of using `pip` directly, we use `poetry` for development purposes to easily track our dependencies.
If you don't have it already, follow the [instructions](https://python-poetry.org/docs/#installation) to install it.

Install the project with dev dependencies:
```bash
poetry install --all-extras
```

### Follow our style

```bash
# install pre-commit hooks
pre-commit install

# apply style and linter checks on staged files
pre-commit
```


## Acknowledgment

gym-xarm is adapted from [FOWM](https://www.yunhaifeng.com/FOWM/) and is based on work by [Nicklas Hansen](https://nicklashansen.github.io/), [Yanjie Ze](https://yanjieze.com/), [Rishabh Jangir](https://jangirrishabh.github.io/), [Mohit Jain](https://natsu6767.github.io/), and [Sambaran Ghosal](https://github.com/SambaranRepo) as part of the following publications:
* [Self-Supervised Policy Adaptation During Deployment](https://arxiv.org/abs/2007.04309)
* [Generalization in Reinforcement Learning by Soft Data Augmentation](https://arxiv.org/abs/2011.13389)
* [Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation](https://arxiv.org/abs/2107.00644)
* [Look Closer: Bridging Egocentric and Third-Person Views with Transformers for Robotic Manipulation](https://arxiv.org/abs/2201.07779)
* [Visual Reinforcement Learning with Self-Supervised 3D Representations](https://arxiv.org/abs/2210.07241)
