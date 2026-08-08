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
    pick_and_place_on_surface,
)
from robolab.core.task.decorators import atomic
from robolab.core.task.task import Task


@atomic
def _compound_success(env, env_id=None):
    """Rubiks cube outside bowl AND orange/lemon still in bowl."""
    return (
        object_outside_of(env, object="rubiks_cube", container="serving_bowl", require_gripper_detached=True)
        & object_in_container(env, object=["orange_01", "lemon_02"], container="serving_bowl", logical="all")
    )


@configclass
class RecoverToyInBowlTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=_compound_success, params={})


@dataclass
class RecoverToyInBowlTask(Task):
    contact_object_list = ["serving_bowl", "orange_01", "lemon_02", "rubiks_cube", "bin_a01", "table"]
    scene = import_scene("lh_recover_toy_in_bowl.usda", contact_object_list)
    terminations = RecoverToyInBowlTerminations
    instruction = {
        "default": "Something in the bowl doesn't belong. Remove it.",
        "vague": "Fix what's wrong.",
        "specific": "The bowl has two fruits and a Rubik's cube. Move the cube to the bin.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "recovery", "common_sense", "anomaly_detection"]
    subtasks = [
        pick_and_place_on_surface(object="rubiks_cube", surface="table", score=1.0),
    ]
    gt_success_checks = [
        {
            "name": "rubiks_cube_removed_from_bowl",
            "predicate": "object_outside_of",
            "object": "rubiks_cube",
            "container": "serving_bowl",
            "require_gripper_detached": True,
        },
    ]
    gt_invariant_checks = [
        {
            "name": "orange_remains_in_bowl",
            "predicate": "object_in_container",
            "object": "orange_01",
            "container": "serving_bowl",
        },
        {
            "name": "lemon_remains_in_bowl",
            "predicate": "object_in_container",
            "object": "lemon_02",
            "container": "serving_bowl",
        },
    ]
