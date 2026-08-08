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


def _fruits_in_bin_distractors_outside(env):
    """Success requires all fruits in bin_a04 AND distractors (serving_bowl, bin_a01) NOT in bin_a04."""
    return (
        object_in_container(env, object=["lemon_02", "lychee01", "lime01", "orange_01"], container="bin_a04", logical="all", require_gripper_detached=True)
        & object_outside_of(env, object="serving_bowl", container="bin_a04")
        & object_outside_of(env, object="bin_a01", container="bin_a04")
    )


@configclass
class InferConsolidateTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_fruits_in_bin_distractors_outside,
        params={},
    )


@dataclass
class InferConsolidateTask(Task):
    contact_object_list = ["serving_bowl", "bin_a01", "bin_a04", "lemon_02", "lychee01", "lime01", "orange_01", "table"]
    scene = import_scene("lh_infer_consolidate.usda", contact_object_list)
    terminations = InferConsolidateTerminations
    instruction = {
        "default": "These all belong together. Move them all where the lemon is.",
        "vague": "Gather them up.",
        "specific": "Collect all four fruits from everywhere and put them all in the large bin where the lemon already is.",
    }
    episode_length_s: int = 180
    attributes = ["reasoning", "inference", "common_sense", "multi_source"]
    subtasks = [
        pick_and_place(object=["lychee01", "lime01"], container="bin_a04", logical="all", score=1.0),
    ]
