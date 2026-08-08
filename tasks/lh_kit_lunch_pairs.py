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


def kit_lunch_pairs_success(env, env_id=None):
    """Success when both bins have correct contents and spam_can remains outside both bins."""
    bins_correct = object_groups_in_containers(
        env,
        groups=[
            {"object": ["lemon_02", "corn_can"], "container": "bin_a02", "require_gripper_detached": True},
            {"object": ["tuna_can", "orange_01"], "container": "bin_a03", "require_gripper_detached": True},
        ],
        env_id=env_id,
    )
    spam_outside_a02 = object_outside_of(
        env,
        object="spam_can",
        container="bin_a02",
        require_gripper_detached=True,
        env_id=env_id,
    )
    spam_outside_a03 = object_outside_of(
        env,
        object="spam_can",
        container="bin_a03",
        require_gripper_detached=True,
        env_id=env_id,
    )
    if env_id is not None:
        return bins_correct and spam_outside_a02 and spam_outside_a03
    return bins_correct & spam_outside_a02 & spam_outside_a03


@configclass
class KitLunchPairsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=kit_lunch_pairs_success,
        params={},
    )


@dataclass
class KitLunchPairsTask(Task):
    contact_object_list = ["bin_a03", "bin_a02", "tuna_can", "orange_01", "corn_can", "lemon_02", "spam_can", "table"]
    scene = import_scene("lh_kit_lunch_pairs.usda", contact_object_list)
    terminations = KitLunchPairsTerminations
    instruction = {
        "default": "One bin has a complete set. Make the other one match.",
        "vague": "Complete the pair.",
        "specific": "One bin has a can and a fruit. The other only has a can. Add the lemon to complete it. Ignore the spam can.",
    }
    episode_length_s: int = 90
    attributes = ["reasoning", "kitting", "common_sense", "distractors"]
    subtasks = [
        pick_and_place(object="lemon_02", container="bin_a02", score=1.0),
    ]
