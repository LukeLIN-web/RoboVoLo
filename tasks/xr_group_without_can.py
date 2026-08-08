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
    pick_and_place,
)
from robolab.core.task.task import Task


def _success(env, env_id=None):
    return (
        object_in_container(env, object=["orange_01", "lemon_02", "lime01"], container="serving_bowl", logical="all", require_gripper_detached=True)
        & object_outside_of(env, object="spam_can", container="serving_bowl")
        & object_outside_of(env, object="avocado01", container="serving_bowl")
        & object_outside_of(env, object="lychee01", container="serving_bowl")
    )


@configclass
class XrGroupWithoutCanTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_success, params={})


@dataclass
class XrGroupWithoutCanTask(Task):
    contact_object_list = ["orange_01", "lemon_02", "lime01", "red_block", "spam_can", "avocado01", "lychee01", "serving_bowl", "table"]
    scene = import_scene("xr_negate_group_without_can.usda", contact_object_list)
    terminations = XrGroupWithoutCanTaskTerminations
    instruction = {
        "default": "Items are arranged in two groups on either side of the block. Put all items from the group that does NOT contain a can into the bowl",
        "vague": "Move the can-free group into the bowl",
        "specific": "The left group has an orange, lemon, and lime. The right group has a spam can, avocado, and lychee. Since the right group contains a can, put the left group (orange, lemon, lime) into the serving bowl.",
    }
    episode_length_s: int = 90
    attributes = ["complex_reference", "semantics", "negation", "spatial"]
    subtasks = [
        pick_and_place(object=["orange_01", "lemon_02", "lime01"], container="serving_bowl", logical="all", score=1.0),
    ]
