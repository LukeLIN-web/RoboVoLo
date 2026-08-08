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
    # Primary: both objects fully in the bowl
    both_in_bowl = object_in_container(env, object=["cream_cheese", "spam_can"], container="serving_bowl", logical="all", require_gripper_detached=True)

    # Alternative: at least one object in the bowl AND both objects stacked
    # (the top object may protrude out of the bowl)
    any_in_bowl = object_in_container(env, object=["cream_cheese", "spam_can"], container="serving_bowl", logical="any", require_gripper_detached=True)
    objects_stacked = stacked(env, objects=["cream_cheese", "spam_can"])

    return (
        (both_in_bowl | (any_in_bowl & objects_stacked))
        & object_outside_of(env, object="lemon_02", container="serving_bowl")
        & object_outside_of(env, object="orange_01", container="serving_bowl")
    )


@configclass
class XrFirstLastTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrFirstLastTask(Task):
    contact_object_list = ['cream_cheese', 'lemon_02', 'orange_01', 'spam_can', 'serving_bowl', 'table']
    scene = import_scene("xr_count_first_last.usda", contact_object_list)
    terminations = XrFirstLastTaskTerminations
    instruction = {'default': 'Four items are in a row. Put the first and the last into the bowl', 'vague': 'Move the ends of the row', 'specific': 'From left to right: cream cheese, lemon, orange, spam can. Put the cream cheese (first) and the spam can (last) into the serving bowl.'}
    episode_length_s: int = 90
    attributes = ['complex_reference', 'semantics', 'counting']
    subtasks = [
        pick_and_place(object=["cream_cheese", "spam_can"], container="serving_bowl", logical="all", score=1.0),
    ]
