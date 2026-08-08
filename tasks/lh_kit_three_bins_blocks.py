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
class KitThreeBinsBlocksTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {"object": "green_block", "container": "bin_a03", "require_gripper_detached": True, "require_contact_with": False},
                {"object": "red_block", "container": "bin_a01", "require_gripper_detached": True, "require_contact_with": False},
                {"object": "blue_block", "container": "bin_a02", "require_gripper_detached": True, "require_contact_with": False},
            ]
        },
    )


@dataclass
class KitThreeBinsBlocksTask(Task):
    contact_object_list = ["bin_a01", "bin_a02", "bin_a03", "red_block", "blue_block", "green_block", "table"]
    scene = import_scene("lh_kit_three_bins_blocks.usda", contact_object_list)
    terminations = KitThreeBinsBlocksTerminations
    instruction = {
        "default": "Each bin should have a block. Fill the empty one.",
        "vague": "Complete the pattern.",
        "specific": "Two bins have blocks and one is empty. Put the green block in the empty bin.",
    }
    episode_length_s: int = 60
    attributes = ["reasoning", "kitting", "common_sense", "pattern_completion"]
    subtasks = [
        pick_and_place(object="green_block", container="bin_a03", score=1.0),
    ]
