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
    pick_and_place,
)
from robolab.core.task.task import Task


def _complete_set_success(env, env_id=None):
    """All single-color cubes in bin AND distractors NOT in bin."""
    return (
        object_in_container(env, object="red_block", container="bin_a03", require_gripper_detached=True)
        & object_in_container(env, object="blue_block", container="bin_a03", require_gripper_detached=True)
        & object_in_container(env, object="yellow_block", container="bin_a03", require_gripper_detached=True)
        & object_in_container(env, object="green_block", container="bin_a03", require_gripper_detached=True)
        & object_outside_of(env, object="rubiks_cube", container="bin_a03")
        & object_outside_of(env, object="lemon_02", container="bin_a03")
    )


@configclass
class InferCompleteSetTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_complete_set_success,
        params={},
    )


@dataclass
class InferCompleteSetTask(Task):
    contact_object_list = ["bin_a03", "red_block", "blue_block", "yellow_block", "green_block", "rubiks_cube", "lemon_02", "table"]
    scene = import_scene("lh_infer_complete_set.usda", contact_object_list)
    terminations = InferCompleteSetTerminations
    instruction = {
        "default": "Complete the collection of items in the container",
        "vague": "Finish the set in the bin.",
        "specific": "The bin has three colored blocks — red, blue, and yellow. The green block on the table completes the set of four. Pick up the green block and place it in the bin.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "inference", "common_sense", "set_completion"]
    subtasks = [
        pick_and_place(object="green_block", container="bin_a03", score=1.0),
    ]
