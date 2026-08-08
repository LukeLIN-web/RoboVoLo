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
from robolab.core.task.decorators import atomic
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@atomic
def _kit_shelf_restock_success(env, env_id=None):
    """BBQ sauce placed on right shelf AND existing bottles remain in place."""
    return (
        object_on_top(env, object=["bbq_sauce_bottle", "ranch_dressing"], reference_object="red_crate_1", logical="all", require_gripper_detached=True)
        & object_on_top(env, object=["ketchup_bottle", "mustard_bottle"], reference_object="red_crate", logical="all", require_gripper_detached=True)
    )


@configclass
class KitShelfRestockTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_kit_shelf_restock_success,
        params={},
    )


@dataclass
class KitShelfRestockTask(Task):
    contact_object_list = ["red_crate", "red_crate_1", "ketchup_bottle", "mustard_bottle", "ranch_dressing", "bbq_sauce_bottle", "table"]
    scene = import_scene("lh_kit_shelf_restock.usda", contact_object_list)
    terminations = KitShelfRestockTerminations
    instruction = {
        "default": "Both shelves should be equally stocked. Add what's needed.",
        "vague": "Even things out.",
        "specific": "The left shelf has two bottles and the right shelf has one. Put the bottle from the table on the right shelf.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "kitting", "common_sense", "counting"]
    subtasks = [
        Subtask(name="place_bottle", conditions=[
            partial(object_on_top, object="bbq_sauce_bottle", reference_object="red_crate_1", require_gripper_detached=True),
        ], logical="all", score=1.0),
    ]
