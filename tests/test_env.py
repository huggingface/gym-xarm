import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import gym_xarm  # noqa: F401

@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        ("XarmLift-v0", "state"),
        ("XarmLift-v0", "pixels"),
        ("XarmLift-v0", "pixels_agent_pos"),
    ],
)
def test_env(env_task, obs_type):
    env = gym.make(f"gym_xarm/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)


if __name__ == "__main__":
    test_env("XarmLift-v0", "pixels_agent_pos")
