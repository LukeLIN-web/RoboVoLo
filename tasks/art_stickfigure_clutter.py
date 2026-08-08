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


# Positions where stick figure body parts must remain
_BODY_PART_POSITIONS = {
    "body": (0.52, 0.0, 0.0114),
    "left_arm": (0.54, 0.05, 0.0114),
    "right_arm": (0.54, -0.05, 0.0114),
    "left_leg": (0.44, 0.02, 0.0114),
    "right_leg": (0.44, -0.02, 0.0114),
}


def _success_check(env, env_id=None):
    """Success: head placed at target AND all stick figure body parts remain unmoved."""
    result = object_at(
        env, object="head", position=(0.6, 0.0, 0.0095),
        tolerance=0.04, require_gripper_detached=True, env_id=env_id,
    )
    for part, pos in _BODY_PART_POSITIONS.items():
        result = result & object_at(env, object=part, position=pos, tolerance=0.03, env_id=env_id)
    return result


@configclass
class ArtStickfigureClutterTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_success_check,
    )


@dataclass
class ArtStickfigureClutterTask(Task):
    contact_object_list = ['table', 'body', 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'head', 'distractor_triangle', 'distractor_square', 'pumpkinsmall', 'ping_pong_paddle', 'extra_rect']
    scene = import_scene("wk_art_stickfigure_clutter.usda", contact_object_list)
    terminations = ArtStickfigureClutterTaskTerminations
    instruction = {'default': 'Complete the stick figure by placing the missing head', 'vague': 'Finish the drawing', 'specific': 'Find the charcoal circle among the clutter and place it at the top of the stick figure where the head should be'}
    episode_length_s: int = 90
    attributes = ["world_knowledge", "art", "spatial_reasoning", "distractors"]
    subtasks = [
        Subtask(
            name="place_piece",
            conditions=partial(
                _success_check,
            ),
            score=1.0,
        ),
    ]
