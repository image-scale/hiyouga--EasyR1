# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import shutil
from typing import Optional, Tuple


CHECKPOINT_TRACKER = "checkpoint_tracker.json"


def find_latest_ckpt(ckpt_dir: str) -> Tuple[Optional[str], Optional[int]]:
    """Find the latest checkpoint in the given directory.

    Args:
        ckpt_dir: Path to the checkpoint directory.

    Returns:
        A tuple of (checkpoint_path, global_step) if found, otherwise (None, None).
    """
    tracker_path = os.path.join(ckpt_dir, CHECKPOINT_TRACKER)
    if not os.path.exists(tracker_path):
        return None, None

    with open(tracker_path, "r") as f:
        tracker_data = json.load(f)

    last_global_step = tracker_data.get("last_global_step")
    if last_global_step is None:
        return None, None

    ckpt_path = os.path.join(ckpt_dir, f"global_step_{last_global_step}")
    if os.path.exists(ckpt_path):
        return ckpt_path, last_global_step

    return None, last_global_step


def remove_obsolete_ckpt(
    ckpt_dir: str,
    global_step: int,
    best_global_step: Optional[int] = None,
    save_limit: int = 3,
) -> None:
    """Remove obsolete checkpoints, keeping only the best and most recent ones.

    Args:
        ckpt_dir: Path to the checkpoint directory.
        global_step: Current global step (will be saved, so reserve a slot for it).
        best_global_step: The global step of the best checkpoint (always kept).
        save_limit: Maximum number of checkpoints to keep (including best and current).
    """
    # Find all global_step_X directories
    pattern = re.compile(r"global_step_(\d+)$")
    ckpt_steps = []

    for name in os.listdir(ckpt_dir):
        match = pattern.match(name)
        if match:
            step = int(match.group(1))
            ckpt_steps.append(step)

    # Determine which checkpoints to keep
    steps_to_keep = set()

    # Always keep the best checkpoint
    if best_global_step is not None and best_global_step in ckpt_steps:
        steps_to_keep.add(best_global_step)

    # Sort remaining steps (excluding best) in descending order
    remaining_steps = sorted([s for s in ckpt_steps if s != best_global_step], reverse=True)

    # Calculate how many more checkpoints we can keep
    # Reserve slots: 1 for best (if exists), 1 for current (global_step)
    slots_used = len(steps_to_keep) + 1  # +1 for the current global_step that will be saved
    slots_available = save_limit - slots_used

    # Keep the most recent checkpoints
    for step in remaining_steps[:max(0, slots_available)]:
        steps_to_keep.add(step)

    # Remove checkpoints that are not in steps_to_keep
    for step in ckpt_steps:
        if step not in steps_to_keep:
            ckpt_path = os.path.join(ckpt_dir, f"global_step_{step}")
            shutil.rmtree(ckpt_path, ignore_errors=True)
