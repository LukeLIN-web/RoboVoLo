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

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from robolab.core.task.task import Task
from robolab.core.task.conditionals import (
    object_in_container,
    object_outside_of_and_on_surface,
    pick_and_place,
)
from robolab.core.task.decorators import atomic
from robolab.core.scenes.utils import import_scene


@atomic
def _cup_clutter_success(env, env_id=None):
    """Coffee cup in trash. Banana and soda can must not end up in trash."""
    correct = object_in_container(
        env,
        object=["coffee_cup"],
        container="trash_bin",
        logical="all",
        require_gripper_detached=True,
    )
    banana_out = object_outside_of_and_on_surface(
        env,
        object=["banana"],
        container="trash_bin",
        surface="table",
        logical="all",
        require_gripper_detached=True,
    )
    can_out = object_outside_of_and_on_surface(
        env,
        object=["soda_can"],
        container="trash_bin",
        surface="table",
        logical="all",
        require_gripper_detached=True,
    )
    return correct & banana_out & can_out


@configclass
class RecycleCupClutterTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_cup_clutter_success, params={})


@dataclass
class RecycleCupClutterTask(Task):
    contact_object_list = [
        "table",
        "compost_bin",
        "trash_bin",
        "recycle_bin",
        "coffee_cup",
        "banana",
        "soda_can",
    ]
    scene = import_scene("wk_recycle_cup_clutter.usda", contact_object_list)
    terminations = RecycleCupClutterTerminations
    instruction = {
        "default": "Throw away only the coffee cup",
        "vague": "Get rid of the cup",
        "specific": "Pick up the coffee cup from the table and place it in the trash bin. Leave the banana and the soda can on the table.",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "recycling", "semantics", "clutter"]
    subtasks = [
        pick_and_place(
            object=["coffee_cup"],
            container="trash_bin",
            logical="all",
            score=1.0,
        ),
    ]
