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
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_groups_in_containers,
    object_in_container,
    object_on_top,
    object_outside_of,
    pick_and_place,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


def _lime_in_bowl_distractors_outside(env):
    """Success requires lime01 in serving_bowl AND distractors (tuna_can, corn_can) NOT in the bowl."""
    return (
        object_in_container(env, object="lime01", container="serving_bowl", require_gripper_detached=True)
        & object_outside_of(env, object="tuna_can", container="serving_bowl")
        & object_outside_of(env, object="corn_can", container="serving_bowl")
    )


@configclass
class InferJoinGroupTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_lime_in_bowl_distractors_outside,
        params={},
    )


@dataclass
class InferJoinGroupTask(Task):
    contact_object_list = ["serving_bowl", "orange_01", "lemon_02", "lime01", "tuna_can", "corn_can", "table"]
    scene = import_scene("lh_infer_join_group.usda", contact_object_list)
    terminations = InferJoinGroupTerminations
    instruction = {
        "default": "One item on the table belongs with the group in the bowl. Put it there. Leave the rest.",
        "vague": "Add what fits.",
        "specific": "The bowl has two fruits. The lime on the table is also a fruit — put it in the bowl. Leave the cans.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "inference", "common_sense", "distractors"]
    subtasks = [
        pick_and_place(object="lime01", container="serving_bowl", score=1.0),
    ]
