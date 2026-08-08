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


@configclass
class InferShelfFallenTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_on_top,
        params={"object": ["ranch_dressing", "ketchup_bottle", "mustard_bottle"], "reference_object": "wireshelving_a01", "logical": "all", "require_gripper_detached": True},
    )


@dataclass
class InferShelfFallenTask(Task):
    contact_object_list = ["wireshelving_a01", "ketchup_bottle", "mustard_bottle", "ranch_dressing", "table"]
    scene = import_scene("lh_infer_shelf_fallen.usda", contact_object_list)
    terminations = InferShelfFallenTerminations
    instruction = {
        "default": "Something fell. Put it back where it belongs.",
        "vague": "Fix it.",
        "specific": "The ranch dressing fell off the shelf. Put it back up with the other bottles.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "inference", "common_sense", "spatial_inference"]
    subtasks = [
        Subtask(name="place_ranch", conditions=[
            partial(object_on_top, object="ranch_dressing", reference_object="wireshelving_a01", require_gripper_detached=True),
        ], logical="all", score=1.0),
    ]
