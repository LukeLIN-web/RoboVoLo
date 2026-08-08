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

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from dataclasses import dataclass
from robolab.core.task.task import Task
from robolab.core.task.conditionals import (
    object_in_container,
    pick_and_place,
)
from robolab.core.scenes.utils import import_scene


@configclass
class RecycleBananaTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["banana"],
            "container": "compost_bin",
            "logical": "all",
            "require_gripper_detached": True,
        },
    )


@dataclass
class RecycleBananaTask(Task):
    contact_object_list = [
        "table",
        "compost_bin",
        "trash_bin",
        "recycle_bin",
        "banana",
    ]
    scene = import_scene("wk_recycle_banana.usda", contact_object_list)
    terminations = RecycleBananaTerminations
    instruction = {
        "default": "Throw away the banana in the correct bin",
        "vague": "Dispose of this properly",
        "specific": "The banana is organic food waste and should go in the beige compost bin labeled 'Compost' with the green sprout icon.",
    }
    episode_length_s: int = 90
    attributes = ["world_knowledge", "recycling", "semantics"]
    subtasks = [
        pick_and_place(
            object=["banana"],
            container="compost_bin",
            logical="all",
            score=1.0,
        ),
    ]
