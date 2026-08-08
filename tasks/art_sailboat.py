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


def _sailboat_success(env, env_id=None):
    """Success: sail placed at either left or right of mast (disjunction)."""
    left = object_at(
        env, object="sail",
        position=(0.55, 0.025, 0.0095), tolerance=0.04,
        require_gripper_detached=True, env_id=env_id,
    )
    right = object_at(
        env, object="sail",
        position=(0.55, -0.045, 0.0095), tolerance=0.04,
        require_gripper_detached=True, env_id=env_id,
    )
    return left | right


@configclass
class ArtSailboatTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_sailboat_success,
    )


@dataclass
class ArtSailboatTask(Task):
    contact_object_list = ['table', 'water', 'hull', 'bow', 'mast', 'sun', 'sail', 'distractor_triangle']
    scene = import_scene("wk_art_sailboat.usda", contact_object_list)
    terminations = ArtSailboatTaskTerminations
    instruction = {'default': 'Install the sail on the boat', 'vague': 'Complete the boat', 'specific': 'Pick up the white triangle and place it next to the mast to be the sail of the boat'}
    episode_length_s: int = 90
    attributes = ["world_knowledge", "art", "spatial_reasoning"]
    subtasks = [
        Subtask(
            name="place_piece",
            conditions=partial(
                _sailboat_success,
            ),
            score=1.0,
        ),
    ]
