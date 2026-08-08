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
    object_grabbed,
    object_groups_in_containers,
    object_in_container,
    object_outside_of,
    pick_and_place,
    pick_and_place_on_surface,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


def _table_items_in_bowl_containers_cleared(env):
    """Objects originally on the table must be in serving_bowl;
    objects originally in containers must NOT be in any container."""
    return (
        object_in_container(env, object="rubiks_cube", container="serving_bowl",
                            require_gripper_detached=True)
        & object_in_container(env, object="banana", container="serving_bowl",
                                require_gripper_detached=True)
        & object_outside_of(env, object="fork_big", container="ceramic_mug")
        & object_outside_of(env, object="lemon_02", container="serving_bowl")
        & object_outside_of(env, object="chocolate_pudding_mix", container="bin_a01")
    )


@configclass
class ThreeContainerRestoreTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_table_items_in_bowl_containers_cleared,
        params={},
    )


@dataclass
class ThreeContainerRestoreTask(Task):
    contact_object_list = [
        "ceramic_mug",
        "serving_bowl",
        "bin_a01",
        "fork_big",
        "lemon_02",
        "chocolate_pudding_mix",
        "rubiks_cube",
        "banana",
        "table",
    ]
    scene = import_scene("lh_three_container_restore.usda", contact_object_list)
    terminations = ThreeContainerRestoreTerminations
    instruction = {
        "default": "Take all objects out of the three containers and place them on the table, then put the objects that were originally on the table into the serving bowl",
        "vague": "Empty all containers onto the table, then gather the items that started on the table into the bowl",
        "specific": "Remove the fork from the ceramic mug, the lemon from the serving bowl, and the chocolate pudding from the bin and place them on the table. Then pick up the rubiks cube and the banana and place both into the serving bowl",
    }
    episode_length_s: int = 240
    attributes = ["memory", "recall", "multi_container"]
    # Two ordered phases.  Within each phase, the per-object work is
    # any-order — collapsed into a single composite that tracks all
    # phase objects in parallel.
    #
    # Score weights are evenly per pick-place op: Phase 1 has 3 target
    # objects, Phase 2 has 2 — 5 ops total → Phase 1 = 3/5 = 0.6,
    # Phase 2 = 2/5 = 0.4.
    #
    # The previous 5-sequential-subtask encoding (clear_mug,
    # clear_bowl, clear_bin, table_item_1, table_item_2) hard-coded an
    # order within each phase even though the success terminator is
    # final-state-only.  A policy that cleared the bowl before the
    # mug was stuck on the SubtaskStateMachine's first subtask
    # (clear_mug) and the score sat at 0 even while the bowl item was
    # being correctly cleared.  The collapse fixes that: ``score`` is
    # the mean per-object completion within each active phase, so any
    # within-phase order produces a monotone score curve.
    #
    # The previous encoding's defensive ``object_outside_of`` guards
    # in Step 5 (which prevented score-credit if a cleared item had
    # been put back into its original container during Phase 2) are
    # dropped here.  ``Terminations.success`` still checks the global
    # state, and SUBTASK_REGRESSION (the score-level path in
    # ``gt_rules.check_subtask_regression``) catches a Phase-1
    # condition reverting during Phase 2 with a fresh event.
    subtasks = [
        # Phase 1: empty all three containers onto the table — any order.
        pick_and_place_on_surface(
            object=["fork_big", "lemon_02", "chocolate_pudding_mix"],
            surface="table",
            logical="all",
            score=0.6,
        ),
        # Phase 2: place the originally-on-table items into the bowl — any order.
        pick_and_place(
            object=["rubiks_cube", "banana"],
            container="serving_bowl",
            logical="all",
            score=0.4,
        ),
    ]
