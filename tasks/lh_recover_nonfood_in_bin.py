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
    """Rubiks cube removed from bin AND all cans remain inside."""
    return (
        object_outside_of(env, object="rubiks_cube", container="bin_a04", require_gripper_detached=True, env_id=env_id)
        & object_in_container(env, object=["corn_can", "tuna_can", "green_beans_can"], container="bin_a04", logical="all", env_id=env_id)
    )


@configclass
class RecoverNonFoodInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_compound_success, params={})


@dataclass
class RecoverNonFoodInBinTask(Task):
    contact_object_list = ["bin_a04", "corn_can", "tuna_can", "green_beans_can", "rubiks_cube", "table"]
    scene = import_scene("lh_recover_nonfood_in_bin.usda", contact_object_list)
    terminations = RecoverNonFoodInBinTerminations
    instruction = {
        "default": "Something in the bin doesn't belong. Take it out.",
        "vague": "Remove the odd one.",
        "specific": "The bin has three cans and a Rubik's cube. The cube doesn't belong. Take it out.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "recovery", "common_sense", "anomaly_detection"]
    subtasks = [
        pick_and_place_on_surface(object="rubiks_cube", surface="table", score=1.0),
    ]
    gt_success_checks = [
        {
            "name": "rubiks_cube_removed_from_bin",
            "predicate": "object_outside_of",
            "object": "rubiks_cube",
            "container": "bin_a04",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "corn_can_remains_in_bin",
            "predicate": "object_in_container",
            "object": "corn_can",
            "container": "bin_a04",
        },
        {
            "name": "tuna_can_remains_in_bin",
            "predicate": "object_in_container",
            "object": "tuna_can",
            "container": "bin_a04",
        },
        {
            "name": "green_beans_can_remains_in_bin",
            "predicate": "object_in_container",
            "object": "green_beans_can",
            "container": "bin_a04",
        },
    ]
