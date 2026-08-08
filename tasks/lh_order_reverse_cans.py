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
    pick_and_place_on_surface,
    stacked,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class ReverseCansStackTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=stacked,
        params={"objects": ["corn_can", "tuna_can"], "order": "bottom_to_top"},
    )


@dataclass
class ReverseCansStackTask(Task):
    contact_object_list = ['tuna_can', 'corn_can', 'table']
    scene = import_scene("lh_cans_stacked.usda", contact_object_list)
    terminations = ReverseCansStackTaskTerminations
    instruction = {
        "default": "Reverse the two-can stack",
        "vague": "Flip the can stack",
        "specific": "Take the top can off and put it on the bottom",
    }
    episode_length_s: int = 60
    attributes = ['memory', 'ordering', 'stacking']
    subtasks = [
        pick_and_place_on_surface(
            object=["corn_can"], surface="table",
            logical="all", score=0.5,
        ),
        Subtask(
            name="reverse",
            conditions=partial(stacked, objects=["corn_can", "tuna_can"], order="bottom_to_top"),
            score=0.5,
        ),
    ]
