from collections import OrderedDict

from gym_xarm.tasks.base import Base as Base
from gym_xarm.tasks.lift import Lift

# from gym_xarm.tasks.peg_in_box import PegInBox
# from gym_xarm.tasks.push import Push
# from gym_xarm.tasks.reach import Reach


TASKS = OrderedDict(
    (
        # (
        #     "reach",
        #     {
        #         "env": Reach,
        #         "action_space": "xyz",
        #         "episode_length": 50,
        #         "description": "Reach a target location with the end effector",
        #     },
        # ),
        # (
        #     "push",
        #     {
        #         "env": Push,
        #         "action_space": "xyz",
        #         "episode_length": 50,
        #         "description": "Push a cube to a target location",
        #     },
        # ),
        # (
        #     "peg_in_box",
        #     {
        #         "env": PegInBox,
        #         "action_space": "xyz",
        #         "episode_length": 50,
        #         "description": "Insert a peg into a box",
        #     },
        # ),
        (
            "lift",
            {
                "env": Lift,
                "action_space": "xyzw",
                "episode_length": 50,
                "description": "Lift a cube above a height threshold",
            },
        ),
    )
)
