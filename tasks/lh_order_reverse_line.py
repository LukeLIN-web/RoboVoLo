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
    object_left_of,
    object_right_of,
)
from robolab.core.task.decorators import atomic
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@atomic
def _reverse_line_success(env, env_id=0, object=None, objects=None):
    """All three items in reversed order: lime01 (left) -> orange_01 (middle) -> lemon_02 (right).

    Accepts and ignores ``object``/``objects`` kwargs so callers can
    pass them via ``partial(_reverse_line_success, object=[…])`` for
    the GTStateExporter's ``object_completed`` lookup (audit Gap 1).
    """
    return (
        object_left_of(env, object="lime01", reference_object="orange_01", require_gripper_detached=True)
        & object_left_of(env, object="orange_01", reference_object="lemon_02", require_gripper_detached=True)
    )


@configclass
class ReverseLineTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_reverse_line_success,
        params={},
    )


@dataclass
class ReverseLineTask(Task):
    contact_object_list = ['lemon_02', 'orange_01', 'lime01', 'table']
    scene = import_scene("lh_three_items_line.usda", contact_object_list)
    terminations = ReverseLineTerminations
    instruction = {
        "default": "Reverse the order of the three items in the line \u2014 the leftmost goes to the right and the rightmost goes to the left",
        "vague": "Flip the line around",
        "specific": "Swap the positions of the left and right items using a temporary buffer spot, leaving the middle item in place",
    }
    episode_length_s: int = 90
    attributes = ['memory', 'ordering', 'spatial']
    # Two-stage scoring with proper per-object targets:
    #   1. ``halfway`` flips when EITHER lime moved to the left of orange
    #      OR lemon moved to the right of orange.  Both objects appear as
    #      group keys so the GT rule sees them as legitimate targets and
    #      doesn't false-positive WRONG_OBJECT_PICKED for whichever the
    #      policy moves first.
    #   2. ``complete`` is the full reversed-line spatial check (also
    #      the termination condition).
    subtasks = [
        Subtask(
            name="halfway",
            conditions={
                "lime01": partial(
                    object_left_of, object="lime01",
                    reference_object="orange_01",
                    require_gripper_detached=True,
                ),
                "lemon_02": partial(
                    object_right_of, object="lemon_02",
                    reference_object="orange_01",
                    require_gripper_detached=True,
                ),
            },
            logical="any",
            score=0.5,
        ),
        Subtask(
            name="complete",
            conditions=partial(
                _reverse_line_success,
                object=["lime01", "orange_01", "lemon_02"],
            ),
            score=0.5,
        ),
    ]
