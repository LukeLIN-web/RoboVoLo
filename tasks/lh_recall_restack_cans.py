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

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import pick_and_place_on_surface
from robolab.core.task.predicate_logic import check_stacked
from robolab.core.task.decorators import atomic
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task
from robolab.core.world.world_state import get_world


# Initial XY position of the tuna_can (bottom of the original stack)
_INITIAL_STACK_XY = (0.50, 0.0)


@atomic
def stacked_and_displaced(
    env,
    objects: list[str],
    order: str = "bottom_to_top",
    base_object: str = "tuna_can",
    initial_xy: tuple[float, float] = _INITIAL_STACK_XY,
    min_displacement: float = 0.10,
    tolerance: float = 0.01,
    env_id: int | None = None,
):
    """Check if objects are stacked in order AND the base object has moved from its initial XY position.

    Handles both the single-env scalar path (``env_id`` int) and the
    vectorized path (``env_id is None`` → poses are ``(num_envs, 3)``).
    """
    world = get_world(env)
    is_stacked = check_stacked(world, objects, order, tolerance, env_id=env_id)
    pos, _ = world.get_pose(base_object, env_id=env_id)
    initial = torch.tensor(initial_xy, device=pos.device, dtype=pos.dtype)
    if env_id is not None:
        # pos: (3,) → compare xy
        displaced = torch.norm(pos[:2] - initial) > min_displacement
        return bool(is_stacked) and bool(displaced)
    # Vectorized: pos (N, 3); is_stacked (N,) bool tensor
    displaced = torch.norm(pos[:, :2] - initial.unsqueeze(0), dim=1) > min_displacement
    return is_stacked & displaced


@configclass
class RecallRestackCansTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=stacked_and_displaced,
        params={
            "objects": ["tuna_can", "corn_can", "tomato_soup_can"],
            "order": "bottom_to_top",
            "base_object": "tuna_can",
            "initial_xy": _INITIAL_STACK_XY,
            "min_displacement": 0.10,
        },
    )


@dataclass
class RecallRestackCansTask(Task):
    contact_object_list = ['tuna_can', 'corn_can', 'tomato_soup_can', 'table']
    scene = import_scene("lh_three_cans_tower.usda", contact_object_list)
    terminations = RecallRestackCansTerminations
    instruction = {
        "default": "Unstack the three cans and place them all on the table. Then rebuild the stack in the same original order at a different spot on the table.",
        "vague": "Take apart the stack and rebuild it elsewhere",
        "specific": "Remove all cans to the table, then stack them again in the same order at a new location",
    }
    episode_length_s: int = 120
    attributes = ['memory', 'recall', 'stacking']
    # Four ordered phases:
    #   1. Unstack top (tomato_soup_can) onto the table.
    #   2. Unstack middle (corn_can) onto the table.
    #   3. Rebuild bottom 2: corn back on tuna at a NEW XY (uses
    #      ``stacked_and_displaced`` so it is False until tuna moves —
    #      otherwise tuna+corn are stacked at episode start and this
    #      would falsely fire SUBGOAL_COMPLETE).
    #   4. Full rebuild: tomato on top, all displaced (= termination).
    subtasks = [
        pick_and_place_on_surface(
            object=["tomato_soup_can"], surface="table",
            logical="all", score=0.25,
        ),
        pick_and_place_on_surface(
            object=["corn_can"], surface="table",
            logical="all", score=0.25,
        ),
        Subtask(
            name="corn_on_tuna_at_new_spot",
            conditions=partial(
                stacked_and_displaced,
                objects=["tuna_can", "corn_can"],
                order="bottom_to_top",
                base_object="tuna_can",
                initial_xy=_INITIAL_STACK_XY,
                min_displacement=0.10,
            ),
            score=0.25,
        ),
        Subtask(
            name="rebuild",
            conditions=partial(
                stacked_and_displaced,
                objects=["tuna_can", "corn_can", "tomato_soup_can"],
                order="bottom_to_top",
                base_object="tuna_can",
                initial_xy=_INITIAL_STACK_XY,
                min_displacement=0.10,
            ),
            score=0.25,
        ),
    ]
