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
    object_in_container,
    object_outside_of,
    object_outside_of_and_on_surface,
    object_groups_in_containers,
    pick_and_place,
)
from robolab.core.task.task import Task


def _success(env, env_id=None):
    return (
        object_in_container(env, object=["lemon_02", "green_block", "orange_01", "blue_block"], container="blackandbrassbowl_small", logical="all", require_gripper_detached=True)
        & object_outside_of(env, object="red_block", container="blackandbrassbowl_small")
    )


@configclass
class XrFourFromRightTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrFourFromRightTask(Task):
    contact_object_list = ['red_block', 'lemon_02', 'green_block', 'orange_01', 'blue_block', 'blackandbrassbowl_small', 'table']
    scene = import_scene("xr_count_four_from_right.usda", contact_object_list)
    terminations = XrFourFromRightTaskTerminations
    instruction = {'default': 'Five items are in a row. Starting from the right, put the first four into the bowl', 'vague': 'Move four from the right end', 'specific': 'From left to right: red block, lemon, green block, orange, blue block. Put the blue block, orange, green block, and lemon (four rightmost) into the brass bowl.'}
    episode_length_s: int = 120
    attributes = ['complex_reference', 'semantics', 'counting']
    subtasks = [
        pick_and_place(object=["lemon_02", "green_block", "orange_01", "blue_block"], container="blackandbrassbowl_small", logical="all", score=1.0),
    ]
