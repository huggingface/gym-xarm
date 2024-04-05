import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        ("XarmLift-v0", "state"),
        ("XarmLift-v0", "pixels"),
        ("XarmLift-v0", "pixels_agent_pos"),
        # TODO(aliberts): Add other tasks
        # ("reach", False, False),
        # ("reach", True, False),
        # ("push", False, False),
        # ("push", True, False),
        # ("peg_in_box", False, False),
        # ("peg_in_box", True, False),
    ],
)
def test_env(env_task, obs_type):
    import gym_xarm  # noqa: F401
    env = gym.make(f"gym_xarm/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped, skip_render_check=True)
    # env.reset()
    # env.render()


if __name__ == "__main__":
    test_env("XarmLift-v0", "pixels_agent_pos")
    # test_env("XarmLift-v0", "state")