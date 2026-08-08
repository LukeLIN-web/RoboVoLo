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
    return object_groups_in_containers(env, groups=[
        {"object": ["lemon_02", "orange_01"], "container": "bowl_015", "logical": "all", "require_gripper_detached": True},
        {"object": ["lime01", "lychee01"], "container": "large_plastic_plate", "logical": "all", "require_gripper_detached": True},
    ])


@configclass
class XrRowSplitTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrRowSplitTask(Task):
    contact_object_list = ['lemon_02', 'orange_01', 'lime01', 'lychee01', 'bowl_015', 'large_plastic_plate', 'table']
    scene = import_scene("xr_spatial_row_split.usda", contact_object_list)
    terminations = XrRowSplitTaskTerminations
    instruction = {'default': 'Four fruits are in a row between two containers. Put the left two fruits into the left container and the right two into the right container', 'vague': 'Split the row of fruits between the containers', 'specific': 'From left to right: lemon, orange, lime, lychee. Put the lemon and orange into the bowl on the left. Put the lime and lychee onto the plate on the right.'}
    episode_length_s: int = 120
    attributes = ['complex_reference', 'semantics', 'spatial']
    subtasks = [
        pick_and_place(object=["lemon_02", "orange_01"], container="bowl_015", logical="all", score=0.5),
        pick_and_place(object=["lime01", "lychee01"], container="large_plastic_plate", logical="all", score=0.5),
    ]
