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
    object_in_container,
    pick_and_place,
    pick_and_place_on_surface,
)
from robolab.core.task.task import Task


@configclass
class UnstackSelectTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["tuna_can", "tomato_soup_can"],
            "container": "bowl",
            "logical": "all",
            "require_gripper_detached": True,
        },
    )


@dataclass
class UnstackSelectTask(Task):
    contact_object_list = ["tuna_can", "corn_can", "tomato_soup_can", "bowl", "table"]
    scene = import_scene("lh_unstack_select.usda", contact_object_list)
    terminations = UnstackSelectTerminations
    instruction = {
        "default": "Unstack all the cans, then put only the top and bottom cans from the original stack into the bowl — leave the middle can on the table",
        "vague": "Unstack the cans and bowl the outer two",
        "specific": "Remove the tomato soup can from the top and place it on the table, then remove the corn can from the middle and leave it on the table, then pick up the tuna can (originally bottom) and the tomato soup can (originally top) and place them both in the bowl",
    }
    episode_length_s: int = 150
    attributes = ["memory", "recall", "stacking"]
    subtasks = [
        # 1. Unstack tomato_soup (top) onto the table.
        pick_and_place_on_surface(
            object=["tomato_soup_can"], surface="table",
            logical="all", score=0.25,
        ),
        # 2. Unstack corn (middle) onto the table.
        pick_and_place_on_surface(
            object=["corn_can"], surface="table",
            logical="all", score=0.25,
        ),
        # 3. Place tuna and tomato_soup into the bowl in any order
        #    (= termination).
        pick_and_place(
            object=["tuna_can", "tomato_soup_can"],
            container="bowl",
            logical="all",
            score=0.5,
        ),
    ]
