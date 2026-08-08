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
        object_in_container(env, object=["lemon_02", "tuna_can"], container="bin_a03", logical="all", require_gripper_detached=True)
        & object_outside_of(env, object="red_block", container="bin_a03")
        & object_outside_of(env, object="green_block", container="bin_a03")
    )


@configclass
class XrExceptBlocksTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrExceptBlocksTask(Task):
    contact_object_list = ['red_block', 'green_block', 'lemon_02', 'tuna_can', 'bin_a03', 'table']
    scene = import_scene("xr_negate_except_blocks.usda", contact_object_list)
    terminations = XrExceptBlocksTaskTerminations
    instruction = {'default': 'Move everything on the table into the bin except the blocks', 'vague': 'Bin everything but the blocks', 'specific': 'Put the lemon and tuna can into the bin. Leave the red and green blocks on the table.'}
    episode_length_s: int = 90
    attributes = ['complex_reference', 'semantics', 'negation']
    subtasks = [
        pick_and_place(object=["lemon_02", "tuna_can"], container="bin_a03", logical="all", score=1.0),
    ]
