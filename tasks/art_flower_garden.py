# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from functools import partial
from robolab.core.task.task import Task
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import object_at
from robolab.core.scenes.utils import import_scene


def art_flower_garden_success(
    env,
    target_object: str,
    target_position: tuple,
    target_tolerance: float,
    painting_objects: list,
    painting_positions: list,
    painting_tolerance: float = 0.04,
    require_gripper_detached: bool = True,
    env_id: int = 0,
):
    """Check that flower3 is placed correctly AND other painting parts are unmoved."""
    # Check target placement
    result = object_at(
        env,
        object=target_object,
        position=target_position,
        tolerance=target_tolerance,
        require_gripper_detached=require_gripper_detached,
        env_id=env_id,
    )
    # Check each painting part is still in place
    for obj, pos in zip(painting_objects, painting_positions):
        obj_ok = object_at(
            env,
            object=obj,
            position=pos,
            tolerance=painting_tolerance,
            env_id=env_id,
        )
        result = result & obj_ok
    return result


@configclass
class ArtFlowerGardenTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=art_flower_garden_success,
        params={
            "target_object": "flower3",
            "target_position": (0.575, 0.12, 0.0095),
            "target_tolerance": 0.04,
            "painting_objects": [
                "stem1",
                "stem2",
                "stem3",
                "flower1",
                "flower2",
                "sun",
                "ground",
                "distractor_triangle",
            ],
            "painting_positions": [
                (0.523, -0.120, 0.011),
                (0.520, 0.0, 0.011),
                (0.520, 0.120, 0.011),
                (0.575, -0.120, 0.012),
                (0.575, 0.0, 0.012),
                (0.760, 0.200, 0.012),
                (0.475, 0.100, 0.011),
                (0.380, -0.250, 0.011),
            ],
            "painting_tolerance": 0.04,
            "require_gripper_detached": True,
        },
    )


@dataclass
class ArtFlowerGardenTask(Task):
    contact_object_list = ['table', 'ground', 'stem1', 'stem2', 'stem3', 'flower1', 'flower2', 'flower3', 'sun', 'distractor_triangle']
    scene = import_scene("wk_art_flower_garden.usda", contact_object_list)
    terminations = ArtFlowerGardenTaskTerminations
    instruction = {'default': 'Complete the garden by placing the missing flower on its stem', 'vague': 'Finish the garden', 'specific': 'Pick up the coral circle and place it on top of the empty stem to complete the third flower'}
    episode_length_s: int = 90
    attributes = ["world_knowledge", "art", "spatial_reasoning", "nature"]
    subtasks = [
        Subtask(
            name="place_piece",
            conditions=partial(
                art_flower_garden_success,
                target_object="flower3",
                target_position=(0.575, 0.12, 0.0095),
                target_tolerance=0.04,
                painting_objects=[
                    "stem1",
                    "stem2",
                    "stem3",
                    "flower1",
                    "flower2",
                    "sun",
                    "ground",
                    "distractor_triangle",
                ],
                painting_positions=[
                    (0.523, -0.120, 0.011),
                    (0.520, 0.0, 0.011),
                    (0.520, 0.120, 0.011),
                    (0.575, -0.120, 0.012),
                    (0.575, 0.0, 0.012),
                    (0.760, 0.200, 0.012),
                    (0.475, 0.100, 0.011),
                    (0.380, -0.250, 0.011),
                ],
                painting_tolerance=0.04,
                require_gripper_detached=True,
            ),
            score=1.0,
        ),
    ]
