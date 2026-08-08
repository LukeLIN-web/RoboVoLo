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
    object_groups_in_containers,
    object_in_container,
    object_on_top,
    object_outside_of,
    pick_and_place_on_surface,
)
from robolab.core.task.decorators import atomic
from robolab.core.task.task import Task


@atomic
def _compound_success(env, env_id=None):
    """Lime removed from container AND all cans remain inside."""
    return (
        object_outside_of(env, object="lime01", container="container_b03", require_gripper_detached=True, env_id=env_id)
        & object_in_container(env, object=["tuna_can", "corn_can", "spam_can"], container="container_b03", logical="all", env_id=env_id)
    )


@configclass
class RecoverFruitAmongCansTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_compound_success, params={})


@dataclass
class RecoverFruitAmongCansTask(Task):
    contact_object_list = ["container_b03", "tuna_can", "corn_can", "spam_can", "lime01", "table"]
    scene = import_scene("lh_recover_fruit_among_cans.usda", contact_object_list)
    terminations = RecoverFruitAmongCansTerminations
    instruction = {
        "default": "One item in the bin doesn't belong with the others. Take it out.",
        "vague": "Remove the odd one out.",
        "specific": "The bin has three cans and one lime. The lime doesn't belong. Remove it.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "recovery", "common_sense", "anomaly_detection"]
    subtasks = [
        pick_and_place_on_surface(object="lime01", surface="table", score=1.0),
    ]
    gt_success_checks = [
        {
            "name": "lime_removed_from_bin",
            "predicate": "object_outside_of",
            "object": "lime01",
            "container": "container_b03",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "tuna_can_remains_in_bin",
            "predicate": "object_in_container",
            "object": "tuna_can",
            "container": "container_b03",
        },
        {
            "name": "corn_can_remains_in_bin",
            "predicate": "object_in_container",
            "object": "corn_can",
            "container": "container_b03",
        },
        {
            "name": "spam_can_remains_in_bin",
            "predicate": "object_in_container",
            "object": "spam_can",
            "container": "container_b03",
        },
    ]
