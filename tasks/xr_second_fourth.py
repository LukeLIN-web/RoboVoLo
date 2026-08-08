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

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_on_top,
    object_outside_of,
    object_outside_of_and_on_surface,
    object_groups_in_containers,
    pick_and_place,
    pick_and_place_on_surface,
)
from robolab.core.task.decorators import atomic
from robolab.core.task.predicate_logic import _not
from robolab.core.task.task import Task


@atomic
def _success(env, env_id=None):
    return (
        object_on_top(env, object=["lemon_02", "lime01"], reference_object="cutting_board_b01", logical="all", require_gripper_detached=True, env_id=env_id)
        & _not(object_on_top(env, object="lychee01", reference_object="cutting_board_b01", env_id=env_id))
        & _not(object_on_top(env, object="orange_01", reference_object="cutting_board_b01", env_id=env_id))
        & _not(object_on_top(env, object="avocado01", reference_object="cutting_board_b01", env_id=env_id))
    )


@configclass
class XrSecondFourthTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrSecondFourthTask(Task):
    contact_object_list = ['lychee01', 'lemon_02', 'orange_01', 'lime01', 'avocado01', 'cutting_board_b01', 'table']
    scene = import_scene("xr_count_second_fourth.usda", contact_object_list)
    terminations = XrSecondFourthTaskTerminations
    instruction = {'default': 'Five fruits are lined up. Put the second and the fourth from the left onto the cutting board', 'vague': 'Pick the even-positioned fruits', 'specific': 'From left to right: lychee, lemon, orange, lime, avocado. Put the lemon (2nd) and the lime (4th) onto the cutting board.'}
    episode_length_s: int = 90
    attributes = ['complex_reference', 'semantics', 'counting']
    subtasks = [
        pick_and_place_on_surface(object=["lemon_02", "lime01"], surface="cutting_board_b01", logical="all", score=1.0),
    ]
