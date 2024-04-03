import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import gym_xarm  # noqa: F401


def test_env():
    env = gym.make("gym_xarm/XarmLift-v0")
    check_env(env.unwrapped, skip_render_check=True)


if __name__ == "__main__":
    test_env()