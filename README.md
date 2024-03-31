# gym-xarm

A gym environment for xArm

<td><img src="http://remicadene.com/assets/gif/simxarm_tdmpc.gif" width="50%" alt="TDMPC policy on xArm env"/></td>


## Acknowledgment

gym-xarm is adapted from [FOWM](https://www.yunhaifeng.com/FOWM/)


## Installation

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n xarm python=3.10
conda activate xarm
```

Install gym-xarm:
```bash
pip install gym-xarm
```


## Contribute

Instead of using `pip` directly, we use `poetry` for development purposes to easily track our dependencies.
If you don't have it already, follow the [instructions](https://python-poetry.org/docs/#installation) to install it.

Install the project with dev dependencies:
```bash
poetry install --with dev
```

### Add dependencies

The equivalent of `pip install some-package` would just be:
```bash
poetry add some-package
```

### Follow our style

```bash
# install pre-commit hooks
pre-commit install

# apply style and linter checks on staged files
pre-commit
```
