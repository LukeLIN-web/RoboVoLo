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
        object_in_container(env, object="lime01", container="bowl", require_gripper_detached=True)
        & object_in_container(env, object="lemon_02", container="bin_a01", require_gripper_detached=True)
        & object_outside_of(env, object="orange_01", container="bowl")
        & object_outside_of(env, object="orange_01", container="bin_a01")
    )


@configclass
class XrLeftmostRightmostTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrLeftmostRightmostTask(Task):
    contact_object_list = ['lime01', 'orange_01', 'lemon_02', 'bowl', 'bin_a01', 'table']
    scene = import_scene("xr_spatial_leftmost_rightmost.usda", contact_object_list)
    terminations = XrLeftmostRightmostTaskTerminations
    instruction = {'default': 'Three fruits are in a row. Put the leftmost one into the bowl and the rightmost one into the bin', 'vague': 'Move the end fruits to the containers', 'specific': 'The lime is on the left, the orange is in the center, and the lemon is on the right. Put the lime in the bowl and the lemon in the bin.'}
    episode_length_s: int = 180
    attributes = ['complex_reference', 'semantics', 'spatial']
    subtasks = [
        pick_and_place(object="lime01", container="bowl", score=0.5),
        pick_and_place(object="lemon_02", container="bin_a01", score=0.5),
    ]
