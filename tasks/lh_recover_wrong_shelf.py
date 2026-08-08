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
    object_on_top,
    object_outside_of,
    pick_and_place_on_surface,
)
from robolab.core.task.decorators import atomic
from robolab.core.task.task import Task


@atomic
def _recover_wrong_shelf_success(env, env_id=None):
    """Tuna can removed from the shelf onto the table AND condiment bottles remain on the shelf."""
    return (
        object_outside_of(env, object="tuna_can", container="wireshelving_a01", require_gripper_detached=True, env_id=env_id)
        & object_on_top(env, object="tuna_can", reference_object="table", require_gripper_detached=True, env_id=env_id)
        & object_on_top(env, object=["ketchup_bottle", "mustard_bottle"], reference_object="wireshelving_a01", logical="all", require_gripper_detached=True, env_id=env_id)
    )


@configclass
class RecoverWrongShelfTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_recover_wrong_shelf_success, params={})


@dataclass
class RecoverWrongShelfTask(Task):
    contact_object_list = ["wireshelving_a01", "ketchup_bottle", "mustard_bottle", "tuna_can", "table"]
    scene = import_scene("lh_recover_wrong_shelf.usda", contact_object_list)
    terminations = RecoverWrongShelfTerminations
    instruction = {
        "default": "Something on the shelf is out of place. Remove it.",
        "vague": "Fix what's wrong.",
        "specific": "The shelf has condiment bottles and a tuna can. The tuna can doesn't belong. Take it off the shelf and put it on the table.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "recovery", "common_sense", "anomaly_detection"]
    subtasks = [
        pick_and_place_on_surface(object="tuna_can", surface="table", score=1.0),
    ]
    gt_success_checks = [
        {
            "name": "tuna_can_removed_from_shelf",
            "predicate": "object_outside_of",
            "object": "tuna_can",
            "container": "wireshelving_a01",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "ketchup_remains_on_shelf",
            "predicate": "object_on_top",
            "object": "ketchup_bottle",
            "reference_object": "wireshelving_a01",
            "require_gripper_detached": True,
        },
        {
            "name": "mustard_remains_on_shelf",
            "predicate": "object_on_top",
            "object": "mustard_bottle",
            "reference_object": "wireshelving_a01",
            "require_gripper_detached": True,
        },
    ]
