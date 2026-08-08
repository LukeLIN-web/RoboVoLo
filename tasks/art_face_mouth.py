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
from functools import partial
from robolab.core.task.task import Task
from robolab.core.task.subtask import Subtask
from robolab.core.task.conditionals import object_at
from robolab.core.scenes.utils import import_scene


def _face_complete_success(env, env_id=None):
    """Combined success: mouth placed AND face parts remain mostly unmoved (4cm tolerance)."""
    mouth_ok = object_at(
        env, object="mouth", position=(0.51, 0.0, 0.0095),
        tolerance=0.04, require_gripper_detached=True, env_id=env_id,
    )
    outline_ok = object_at(
        env, object="face_outline", position=(0.551, 0.0, 0.0115),
        tolerance=0.04, env_id=env_id,
    )
    left_eye_ok = object_at(
        env, object="left_eye", position=(0.577, 0.027, 0.0115),
        tolerance=0.04, env_id=env_id,
    )
    right_eye_ok = object_at(
        env, object="right_eye", position=(0.582, -0.024, 0.0115),
        tolerance=0.04, env_id=env_id,
    )
    nose_ok = object_at(
        env, object="nose", position=(0.539, -0.002, 0.0114),
        tolerance=0.04, env_id=env_id,
    )
    return mouth_ok & outline_ok & left_eye_ok & right_eye_ok & nose_ok


@configclass
class ArtFaceMouthTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=_face_complete_success,
        params={},
    )


@dataclass
class ArtFaceMouthTask(Task):
    contact_object_list = ['table', 'face_outline', 'left_eye', 'right_eye', 'nose', 'mouth', 'distractor_long', 'distractor_blue']
    scene = import_scene("wk_art_face_mouth.usda", contact_object_list)
    terminations = ArtFaceMouthTaskTerminations
    instruction = {'default': 'Complete the face by placing the missing mouth', 'vague': 'Finish the face', 'specific': 'Pick up the small coral rectangle and place it below the nose to complete the face as the mouth'}
    episode_length_s: int = 90
    attributes = ["world_knowledge", "art", "spatial_reasoning"]
    subtasks = [
        Subtask(
            name="place_piece",
            conditions=partial(
                _face_complete_success,
            ),
            score=1.0,
        ),
    ]
