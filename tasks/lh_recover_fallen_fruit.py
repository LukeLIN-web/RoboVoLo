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
    object_outside_of_and_on_surface,
    pick_and_place,
)
from robolab.core.task.decorators import atomic
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@atomic
def _compound_success(env, env_id=None):
    """Orange in bowl, lime and lemon remain in bowl, AND all cans remain outside bowl on the table."""
    return (
        object_in_container(env, object="orange_01", container="bowl", require_gripper_detached=True, env_id=env_id)
        & object_in_container(env, object="lemon_02", container="bowl", require_gripper_detached=True, env_id=env_id)
        & object_in_container(env, object="lime01", container="bowl", require_gripper_detached=True, env_id=env_id)
        & object_outside_of_and_on_surface(
            env, object=["spam_can", "corn_can", "tomato_soup_can", "green_beans_can"],
            container="bowl", surface="table", logical="all", require_gripper_detached=True, env_id=env_id,
        )
    )


@configclass
class RecoverFallenFruitTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_compound_success,
        params={},
    )


@dataclass
class RecoverFallenFruitTask(Task):
    contact_object_list = ["bowl", "lemon_02", "lime01", "orange_01", "spam_can", "corn_can", "tomato_soup_can", "green_beans_can", "table"]
    scene = import_scene("lh_recover_fallen_fruit.usda", contact_object_list)
    terminations = RecoverFallenFruitTerminations
    instruction = {
        "default": "an object that belongs in the bowl has fallen out. Put it back",
        "vague": "Fix it.",
        "specific": "The orange rolled out of the fruit bowl. Put it back in without disturbing the cans.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "recovery", "common_sense", "spatial_inference"]
    subtasks = [
        pick_and_place(object="orange_01", container="bowl", score=1.0),
    ]
    gt_success_checks = [
        {
            "name": "orange_returned_to_bowl",
            "predicate": "object_in_container",
            "object": "orange_01",
            "container": "bowl",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "lemon_remains_in_bowl",
            "predicate": "object_in_container",
            "object": "lemon_02",
            "container": "bowl",
            "require_gripper_detached": True,
        },
        {
            "name": "lime_remains_in_bowl",
            "predicate": "object_in_container",
            "object": "lime01",
            "container": "bowl",
            "require_gripper_detached": True,
        },
        {
            "name": "cans_remain_outside_bowl_on_table",
            "predicate": "object_outside_of_and_on_surface",
            "object": [
                "spam_can",
                "corn_can",
                "tomato_soup_can",
                "green_beans_can",
            ],
            "container": "bowl",
            "surface": "table",
            "logical": "all",
            "require_gripper_detached": True,
        },
    ]
