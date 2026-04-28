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

from typing import List, Tuple

import torch

from verl.protocol import DataProto


def prepare_dynamic_batch(
    dataproto: DataProto, max_token_len: int
) -> Tuple[List[DataProto], List[List[int]]]:
    """Split a DataProto into micro-batches based on token count constraints.

    Groups samples into micro-batches such that the total token count in each
    micro-batch doesn't exceed max_token_len.

    Args:
        dataproto: The DataProto to split, must contain "attention_mask" key.
        max_token_len: Maximum total tokens per micro-batch.

    Returns:
        A tuple of (micro_batches, micro_bsz_idx_lst) where:
        - micro_batches: List of DataProto, each containing a subset of samples
        - micro_bsz_idx_lst: List of lists, recording which original indices
          are in each micro-batch
    """
    attention_mask = dataproto.batch["attention_mask"]
    batch_size = attention_mask.shape[0]

    # Calculate token count per sample (sum of attention mask)
    token_counts = attention_mask.sum(dim=1).tolist()

    # Group samples into micro-batches
    micro_batches = []
    micro_bsz_idx_lst = []

    current_batch_indices = []
    current_token_count = 0

    for idx in range(batch_size):
        sample_token_count = token_counts[idx]

        # If adding this sample would exceed max_token_len, start a new batch
        if current_token_count + sample_token_count > max_token_len and current_batch_indices:
            # Save current batch
            micro_batches.append(dataproto[current_batch_indices])
            micro_bsz_idx_lst.append(current_batch_indices)

            # Start new batch
            current_batch_indices = []
            current_token_count = 0

        # Add sample to current batch
        current_batch_indices.append(idx)
        current_token_count += sample_token_count

    # Don't forget the last batch
    if current_batch_indices:
        micro_batches.append(dataproto[current_batch_indices])
        micro_bsz_idx_lst.append(current_batch_indices)

    return micro_batches, micro_bsz_idx_lst


def restore_dynamic_batch(
    tensor: torch.Tensor, micro_bsz_idx_lst: List[List[int]]
) -> torch.Tensor:
    """Restore a tensor to the original order after dynamic batching.

    Args:
        tensor: The concatenated tensor from micro-batches.
        micro_bsz_idx_lst: The list of original indices for each micro-batch.

    Returns:
        The tensor restored to original sample order.
    """
    # Flatten the index list and create inverse mapping
    flat_indices = []
    for idx_lst in micro_bsz_idx_lst:
        flat_indices.extend(idx_lst)

    # Create inverse permutation: for each original index, where is it in the flattened order?
    total_size = len(flat_indices)
    inverse_indices = [0] * total_size

    for new_idx, original_idx in enumerate(flat_indices):
        inverse_indices[original_idx] = new_idx

    # Reorder tensor back to original order
    return tensor[inverse_indices]
