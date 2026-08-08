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
)
from robolab.core.task.decorators import atomic
from robolab.core.task.predicate_logic import in_contact, left_of
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task
from robolab.core.world.world_state import get_world


def _and(a, b):
    """Logical AND that works for both bool and Tensor."""
    return a & b if not isinstance(a, bool) else a and b


@atomic
def line_up_red_blue_green(env, env_id=None, object=None, objects=None):
    """Success: red, blue, green arranged left-to-right, all resting on the table, gripper released.

    Composed explicitly because object_left_of skips its level check when
    require_gripper_detached=True, which lets a still-stacked-but-nudged tower
    pass the cone-only ordering test. We replace that with an explicit
    is_supported_on_surface(block, "table") check, which fails for any block
    resting on another block.

    Accepts and ignores ``object``/``objects`` kwargs so callers can
    pass them via ``partial(line_up_red_blue_green, object=[…])`` for
    the GTStateExporter's ``object_completed`` lookup (audit Gap 1).
    """
    world = get_world(env)
    blocks = ["red_block", "blue_block", "green_block"]

    result = left_of(world, "red_block", "blue_block")
    result = _and(result, left_of(world, "blue_block", "green_block"))

    for block in blocks:
        result = _and(result, world.is_supported_on_surface(block, "table"))
        contact = in_contact(world, block, "gripper")
        not_contact = ~contact if not isinstance(contact, bool) else not contact
        result = _and(result, not_contact)

    return result


@configclass
class LineUpByStackOrderTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=line_up_red_blue_green,
        params={},
    )


@dataclass
class LineUpByStackOrderTask(Task):
    contact_object_list = ['red_block', 'blue_block', 'green_block', 'table']
    scene = import_scene("lh_tower_rbg_bin.usda", contact_object_list)
    terminations = LineUpByStackOrderTerminations
    instruction = {
        "default": "The three blocks are stacked in a tower. Unstack them all onto the table. Then line them up from left to right in their original stacking order \u2014 the one that was on the bottom goes on the left, the one from the top goes on the right.",
        "vague": "Unstack and line up by original position",
        "specific": "Take all blocks off the tower, then arrange them left to right matching the bottom-to-top order of the original stack",
    }
    episode_length_s: int = 120
    attributes = ['memory', 'ordering', 'spatial']
    subtasks = [
        # 1. Unstack top (green) to table — physics forces top-down order.
        pick_and_place_on_surface(
            object=["green_block"], surface="table",
            logical="all", score=1 / 3,
        ),
        # 2. Unstack middle (blue) to table.
        pick_and_place_on_surface(
            object=["blue_block"], surface="table",
            logical="all", score=1 / 3,
        ),
        # 3. Final line-up (= termination).  ``object=[…]`` is read by
        # GTStateExporter._export_object_completed so per-block completion
        # is reported under real names instead of a generic placeholder.
        Subtask(
            name="arrange",
            conditions=partial(
                line_up_red_blue_green,
                object=["red_block", "blue_block", "green_block"],
            ),
            score=1 / 3,
        ),
    ]
