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
        object_in_container(env, object="spam_can", container="serving_bowl", require_gripper_detached=True)
        & object_outside_of(env, object="corn_can", container="serving_bowl")
        & object_outside_of(env, object="bin_a01", container="serving_bowl")
        & object_outside_of(env, object="bin_a02", container="serving_bowl")
    )


@configclass
class XrBetweenBinsTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrBetweenBinsTask(Task):
    contact_object_list = ['spam_can', 'corn_can', 'serving_bowl', 'bin_a01', 'bin_a02', 'table']
    scene = import_scene("xr_spatial_between_bins_v2.usda", contact_object_list)
    terminations = XrBetweenBinsTaskTerminations
    instruction = {'default': 'Put only the item that is between the two bins into the bowl', 'vague': "Move only what's between the bins", 'specific': 'The spam can is positioned between the two bins. Pick it up and place only it into the serving bowl.'}
    episode_length_s: int = 90
    attributes = ['complex_reference', 'semantics', 'spatial']
    subtasks = [
        pick_and_place(object="spam_can", container="serving_bowl", score=1.0),
    ]
