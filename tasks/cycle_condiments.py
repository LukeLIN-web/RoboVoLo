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
from robolab.core.task.conditionals import object_left_of
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


def cycled_arrangement(env, env_id=None, object=None, objects=None):
    """Custom compound termination: BBQ sauce left-of ketchup AND ketchup left-of mustard.
    Initial: ketchup(left), mustard(center), bbq(right)
    Target:  bbq(left), ketchup(center), mustard(right)

    Accepts and ignores ``object``/``objects`` kwargs so callers can
    pass them via ``partial(cycled_arrangement, object=[…])`` for the
    GTStateExporter's ``object_completed`` lookup (audit Gap 1).
    """
    cond1 = object_left_of(env, object="bbq_sauce_bottle", reference_object="ketchup_bottle", require_gripper_detached=True, env_id=env_id)
    cond2 = object_left_of(env, object="ketchup_bottle", reference_object="mustard_bottle", require_gripper_detached=True, env_id=env_id)
    if env_id is not None:
        return cond1 and cond2
    return cond1 & cond2


@configclass
class CycleCondimentsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=cycled_arrangement,
        params={},
    )


@dataclass
class CycleCondimentsTask(Task):
    contact_object_list = ["ketchup_bottle", "mustard_bottle", "bbq_sauce_bottle", "table"]
    scene = import_scene("lh_cycle_condiments.usda", contact_object_list)
    terminations = CycleCondimentsTerminations
    instruction = {
        "default": "Rotate each bottle one position to the right — the rightmost bottle wraps around to the left position",
        "vague": "Shift the bottles one spot to the right in a cycle",
        "specific": "Move the BBQ sauce bottle from the right to the left position, the ketchup bottle from the left to the center position, and the mustard bottle from the center to the right position, using a temporary spot on the table as needed",
    }
    episode_length_s: int = 180
    attributes = ["memory", "ordering", "spatial"]
    # Single subtask matching termination — the spatial-cycle check.
    # ``object=[…]`` is consumed by GTStateExporter._export_object_completed
    # so all three bottles are exposed as task targets.
    subtasks = [
        Subtask(
            name="cycle_complete",
            conditions=partial(
                cycled_arrangement,
                object=["bbq_sauce_bottle", "ketchup_bottle", "mustard_bottle"],
            ),
            score=1.0,
        ),
    ]
