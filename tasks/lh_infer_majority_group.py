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
class InferMajorityGroupTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={"groups": [
            {"object": ["lemon_02", "lime01", "lychee01"], "container": "bowl", "logical": "all", "require_gripper_detached": True, "require_contact_with": False},
        ]},
    )


@dataclass
class InferMajorityGroupTask(Task):
    contact_object_list = ["bowl", "lemon_02", "lime01", "lychee01", "tuna_can", "table"]
    scene = import_scene("lh_infer_majority_group.usda", contact_object_list)
    terminations = InferMajorityGroupTerminations
    instruction = {
        "default": "Most items on the table are the same kind. Put them in the bowl and leave the odd one out.",
        "vague": "Group the majority.",
        "specific": "There are three fruits and one can. Put all three fruits in the bowl. Leave the tuna can on the table.",
    }
    episode_length_s: int = 120
    attributes = ["reasoning", "inference", "common_sense", "majority_rule"]
    subtasks = [
        pick_and_place(object=["lemon_02", "lime01", "lychee01"], container="bowl", logical="all", score=1.0),
    ]
