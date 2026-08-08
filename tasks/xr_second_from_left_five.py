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
        object_in_container(env, object="spam_can", container="plastic_basket", require_gripper_detached=True)
        & object_outside_of(env, object="red_block", container="plastic_basket")
        & object_outside_of(env, object="lemon_02", container="plastic_basket")
        & object_outside_of(env, object="corn_can", container="plastic_basket")
        & object_outside_of(env, object="orange_01", container="plastic_basket")
    )


@configclass
class XrSecondFromLeftFiveTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrSecondFromLeftFiveTask(Task):
    contact_object_list = ['red_block', 'spam_can', 'lemon_02', 'corn_can', 'orange_01', 'plastic_basket', 'table']
    scene = import_scene("xr_count_second_from_left_five.usda", contact_object_list)
    terminations = XrSecondFromLeftFiveTaskTerminations
    instruction = {'default': 'Five items are lined up. Put only the second one from the left into the basket', 'vague': 'Move the second from the left into the basket', 'specific': 'From left to right: red block, spam can, lemon, corn can, orange. The second from the left is the spam can. Pick up only the spam can and place it in the basket.'}
    episode_length_s: int = 90
    attributes = ['complex_reference', 'semantics', 'counting']
    subtasks = [
        pick_and_place(object="spam_can", container="plastic_basket", score=1.0),
    ]
