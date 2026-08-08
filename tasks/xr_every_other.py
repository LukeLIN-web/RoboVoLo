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
    stacked,
)
from robolab.core.task.task import Task


def _success(env, env_id=None):
    # Distractors must remain outside the bin
    distractors_ok = (
        object_outside_of(env, object="lemon_02", container="bin_a03")
        & object_outside_of(env, object="orange_01", container="bin_a03")
    )
    # Primary: all blocks detected inside the bin
    all_in_bin = object_in_container(env, object=["red_block", "green_block", "blue_block"], container="bin_a03", logical="all", require_gripper_detached=True)
    # Alternative: at least one block in the bin AND all blocks stacked (any order).
    # This handles the case where blocks are stacked in the bin and the top one extends above the rim.
    any_in_bin = object_in_container(env, object=["red_block", "green_block", "blue_block"], container="bin_a03", logical="any", require_gripper_detached=True)
    all_stacked = stacked(env, objects=["red_block", "green_block", "blue_block"], order=None)
    return distractors_ok & (all_in_bin | (any_in_bin & all_stacked))


@configclass
class XrEveryOtherTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrEveryOtherTask(Task):
    contact_object_list = ['red_block', 'lemon_02', 'green_block', 'orange_01', 'blue_block', 'bin_a03', 'table']
    scene = import_scene("xr_count_every_other.usda", contact_object_list)
    terminations = XrEveryOtherTaskTerminations
    instruction = {'default': 'Five items are in a row. Move every other one into the bin, starting from the first on the left', 'vague': 'Move alternate items starting from the left', 'specific': 'From left to right: red block, lemon, green block, orange, blue block. Move the 1st, 3rd, and 5th items (the three blocks) into the bin.'}
    episode_length_s: int = 90
    attributes = ['complex_reference', 'semantics', 'counting']
    subtasks = [
        pick_and_place(object=["red_block", "green_block", "blue_block"], container="bin_a03", logical="all", score=1.0),
    ]
