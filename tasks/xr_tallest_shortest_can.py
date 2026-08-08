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
        object_in_container(env, object="tomato_soup_can", container="serving_bowl", require_gripper_detached=True)
        & object_in_container(env, object="tuna_can", container="open_cardboard_box_002", require_gripper_detached=True)
        & object_outside_of(env, object="corn_can", container="serving_bowl")
        & object_outside_of(env, object="corn_can", container="open_cardboard_box_002")
        & object_outside_of(env, object="spam_can", container="serving_bowl")
        & object_outside_of(env, object="spam_can", container="open_cardboard_box_002")
    )


@configclass
class XrTallestShortestCanTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrTallestShortestCanTask(Task):
    contact_object_list = ['tomato_soup_can', 'corn_can', 'spam_can', 'tuna_can', 'serving_bowl', 'open_cardboard_box_002', 'table']
    scene = import_scene("xr_size_tallest_shortest_can.usda", contact_object_list)
    terminations = XrTallestShortestCanTaskTerminations
    instruction = {'default': 'Put the tallest can into the bowl and the shortest can into the box', 'vague': 'Sort the extreme-height cans into containers', 'specific': 'The tomato soup can is the tallest and the tuna can is the shortest. Put the soup can in the bowl and the tuna can in the box.'}
    episode_length_s: int = 180
    attributes = ['complex_reference', 'semantics', 'size_reasoning']
    subtasks = [
        pick_and_place(object="tomato_soup_can", container="serving_bowl", score=0.5),
        pick_and_place(object="tuna_can", container="open_cardboard_box_002", score=0.5),
    ]
