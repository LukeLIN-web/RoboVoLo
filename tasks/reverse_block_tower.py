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
class ReverseBlockTowerTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=stacked,
        params={
            "objects": ["green_block", "blue_block", "red_block"],
            "order": "bottom_to_top",
        },
    )


@dataclass
class ReverseBlockTowerTask(Task):
    contact_object_list = ["red_block", "blue_block", "green_block", "table"]
    scene = import_scene("lh_reverse_block_tower.usda", contact_object_list)
    terminations = ReverseBlockTowerTerminations
    instruction = {
        "default": "Reverse the stacking order of the three colored blocks",
        "vague": "Flip the block tower order",
        "specific": "Unstack all three blocks and rebuild the tower with green on the bottom, blue in the middle, and red on top",
    }
    episode_length_s: int = 180
    attributes = ["memory", "ordering", "stacking"]
    # Score weights are evenly per pick-place op.  Each step is one
    # block placement (green to table, blue on green, red on blue) —
    # the ``stacked(objects=[a, b, c])`` predicate's ``objects=`` list
    # length describes the cumulative arrangement size, NOT the number
    # of placements the policy performs in that step.  3 ops total →
    # 1/3 per step.
    subtasks = [
        # Step 1: Remove green block from top to table.  Required —
        # green must be off the original tower before it can become
        # the bottom of the new tower.  Encoded as
        # ``pick_and_place_on_surface`` (rather than the previous inline
        # ``[grabbed, on_top(table)]`` list-of-2-partials) so the CSM
        # exposes per-object ``condition_idx 0/1/2/3`` for green_block
        # — that lets WRONG_TARGET_PLACE / WRONG_OBJECT_PICKED
        # suppression work during the unstack phase.
        pick_and_place_on_surface(
            object=["green_block"],
            surface="table",
            logical="all",
            score=1 / 3,
        ),
        # Step 2: Stack blue on green.
        # The previous encoding had an extra "unstack_blue to table"
        # subtask between Step 1 and this one, which forced the policy
        # to buffer blue on the table.  That blocked score progression
        # for any policy that takes blue directly from on-top-of-red
        # onto green (a perfectly valid + more efficient path) — the
        # sequential SM stayed stuck on the buffer subtask and never
        # advanced to read this stack-check, even though the stacked
        # condition was already True.  Removed; the buffer move is
        # policy strategy, not part of the success contract.
        Subtask(
            name="stack_blue_on_green",
            conditions=partial(stacked, objects=["green_block", "blue_block"], order="bottom_to_top"),
            score=1 / 3,
        ),
        # Step 3: Stack red on blue (full reversed tower).
        Subtask(
            name="stack_red_on_blue",
            conditions=partial(stacked, objects=["green_block", "blue_block", "red_block"], order="bottom_to_top"),
            score=1 / 3,
        ),
    ]
