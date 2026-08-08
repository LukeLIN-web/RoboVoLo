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
class InferReturnStrayTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "tuna_can", "container": "bin_a03", "require_gripper_detached": True},
    )


@dataclass
class InferReturnStrayTask(Task):
    contact_object_list = ["bin_a03", "serving_bowl", "corn_can", "spam_can", "orange_01", "lemon_02", "tuna_can", "rubiks_cube", "table"]
    scene = import_scene("lh_infer_return_stray.usda", contact_object_list)
    terminations = InferReturnStrayTerminations
    instruction = {
        "default": "One of the items on the table is out of place. Put it where it belongs.",
        "vague": "Return the stray.",
        "specific": "The tuna can belongs with the other cans in the bin. The Rubik's cube has no container. Put the tuna can in the bin.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "inference", "common_sense", "selective_action"]
    subtasks = [
        pick_and_place(object="tuna_can", container="bin_a03", score=1.0),
    ]
