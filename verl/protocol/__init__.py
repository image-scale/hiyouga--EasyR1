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

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class DataProto:
    """Data protocol for storing batched data with tensors and non-tensors."""

    def __init__(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        non_tensor_batch: Optional[Dict[str, np.ndarray]] = None,
        meta_info: Optional[Dict[str, Any]] = None,
    ):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info or {}

    def __len__(self) -> int:
        if self.batch is not None and len(self.batch) > 0:
            first_key = next(iter(self.batch))
            return self.batch[first_key].shape[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            first_key = next(iter(self.non_tensor_batch))
            return len(self.non_tensor_batch[first_key])
        return 0

    def __getitem__(self, key: Union[int, slice, List[int], torch.Tensor]) -> "DataProto":
        # Convert key to appropriate indexing format
        if isinstance(key, int):
            index = key
        elif isinstance(key, slice):
            index = key
        elif isinstance(key, list):
            index = key
        elif isinstance(key, torch.Tensor):
            index = key.tolist()
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

        new_batch = None
        if self.batch is not None:
            new_batch = {}
            for k, v in self.batch.items():
                new_batch[k] = v[index]

        new_non_tensor_batch = None
        if self.non_tensor_batch is not None:
            new_non_tensor_batch = {}
            for k, v in self.non_tensor_batch.items():
                new_non_tensor_batch[k] = v[index]

        return DataProto(
            batch=new_batch,
            non_tensor_batch=new_non_tensor_batch,
            meta_info=self.meta_info.copy() if self.meta_info else None,
        )

    @classmethod
    def from_dict(
        cls,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        non_tensors: Optional[Dict[str, np.ndarray]] = None,
        meta_info: Optional[Dict[str, Any]] = None,
        num_batch_dims: int = 1,
    ) -> "DataProto":
        # Validate that all tensors have the same batch size
        if tensors is not None and len(tensors) > 0:
            batch_sizes = set()
            for key, tensor in tensors.items():
                # For num_batch_dims > 1, require strictly more dimensions
                # (i.e., at least one feature dimension beyond batch dims)
                if num_batch_dims > 1:
                    assert tensor.ndim > num_batch_dims, (
                        f"Tensor {key} has {tensor.ndim} dims, expected > {num_batch_dims}"
                    )
                batch_sizes.add(tensor.shape[:num_batch_dims])

            assert len(batch_sizes) == 1, f"Inconsistent batch sizes: {batch_sizes}"

        return cls(batch=tensors, non_tensor_batch=non_tensors, meta_info=meta_info)

    @classmethod
    def from_single_dict(cls, data: Dict[str, Any]) -> "DataProto":
        """Create DataProto from a dict where all values are tensors."""
        tensors = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                tensors[k] = v
            else:
                tensors[k] = torch.tensor(v)
        return cls(batch=tensors, non_tensor_batch=None, meta_info=None)

    @classmethod
    def concat(cls, data_list: List["DataProto"]) -> "DataProto":
        """Concatenate multiple DataProtos along the batch dimension."""
        if not data_list:
            return cls()

        # Concatenate batches
        new_batch = None
        if data_list[0].batch is not None:
            new_batch = {}
            for key in data_list[0].batch.keys():
                new_batch[key] = torch.cat([d.batch[key] for d in data_list], dim=0)

        # Concatenate non-tensor batches
        new_non_tensor_batch = None
        if data_list[0].non_tensor_batch is not None:
            new_non_tensor_batch = {}
            for key in data_list[0].non_tensor_batch.keys():
                new_non_tensor_batch[key] = np.concatenate(
                    [d.non_tensor_batch[key] for d in data_list], axis=0
                )

        # Use meta_info from the first DataProto
        meta_info = data_list[0].meta_info.copy() if data_list[0].meta_info else None

        return cls(batch=new_batch, non_tensor_batch=new_non_tensor_batch, meta_info=meta_info)

    @classmethod
    def load_from_disk(cls, path: str) -> "DataProto":
        """Load DataProto from disk."""
        data = torch.load(path, weights_only=False)
        return cls(
            batch=data.get("batch"),
            non_tensor_batch=data.get("non_tensor_batch"),
            meta_info=data.get("meta_info"),
        )

    def save_to_disk(self, path: str) -> None:
        """Save DataProto to disk."""
        data = {
            "batch": self.batch,
            "non_tensor_batch": self.non_tensor_batch,
            "meta_info": self.meta_info,
        }
        torch.save(data, path)

    def select(
        self,
        batch_keys: Optional[List[str]] = None,
        non_tensor_batch_keys: Optional[List[str]] = None,
        meta_info_keys: Optional[List[str]] = None,
    ) -> "DataProto":
        """Select specific keys and return a new DataProto (non-destructive)."""
        new_batch = None
        if batch_keys is not None and self.batch is not None:
            new_batch = {k: self.batch[k] for k in batch_keys if k in self.batch}

        new_non_tensor_batch = None
        if non_tensor_batch_keys is not None and self.non_tensor_batch is not None:
            new_non_tensor_batch = {
                k: self.non_tensor_batch[k] for k in non_tensor_batch_keys if k in self.non_tensor_batch
            }

        new_meta_info = None
        if meta_info_keys is not None and self.meta_info is not None:
            new_meta_info = {k: self.meta_info[k] for k in meta_info_keys if k in self.meta_info}

        return DataProto(
            batch=new_batch,
            non_tensor_batch=new_non_tensor_batch,
            meta_info=new_meta_info,
        )

    def pop(
        self,
        batch_keys: Optional[List[str]] = None,
        non_tensor_batch_keys: Optional[List[str]] = None,
        meta_info_keys: Optional[List[str]] = None,
    ) -> "DataProto":
        """Pop specific keys and return them in a new DataProto (destructive)."""
        popped_batch = None
        if batch_keys is not None and self.batch is not None:
            popped_batch = {}
            for k in batch_keys:
                if k in self.batch:
                    popped_batch[k] = self.batch.pop(k)

        popped_non_tensor_batch = None
        if non_tensor_batch_keys is not None and self.non_tensor_batch is not None:
            popped_non_tensor_batch = {}
            for k in non_tensor_batch_keys:
                if k in self.non_tensor_batch:
                    popped_non_tensor_batch[k] = self.non_tensor_batch.pop(k)

        popped_meta_info = None
        if meta_info_keys is not None and self.meta_info is not None:
            popped_meta_info = {}
            for k in meta_info_keys:
                if k in self.meta_info:
                    popped_meta_info[k] = self.meta_info.pop(k)

        return DataProto(
            batch=popped_batch,
            non_tensor_batch=popped_non_tensor_batch,
            meta_info=popped_meta_info,
        )

    def chunk(self, chunks: int) -> List["DataProto"]:
        """Split into chunks equal-sized chunks."""
        length = len(self)
        assert length % chunks == 0, f"Cannot chunk {length} items into {chunks} equal chunks"

        chunk_size = length // chunks
        result = []

        for i in range(chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            result.append(self[start:end])

        return result

    def split(self, split_size: int) -> List["DataProto"]:
        """Split into chunks of given size."""
        length = len(self)
        result = []

        for i in range(0, length, split_size):
            end = min(i + split_size, length)
            result.append(self[i:end])

        return result

    def reorder(self, indices: torch.Tensor) -> None:
        """Reorder the data in-place according to indices."""
        indices_list = indices.tolist()

        if self.batch is not None:
            for key in self.batch:
                self.batch[key] = self.batch[key][indices_list]

        if self.non_tensor_batch is not None:
            for key in self.non_tensor_batch:
                self.non_tensor_batch[key] = self.non_tensor_batch[key][indices_list]

    def repeat(self, repeat_times: int, interleave: bool = False) -> "DataProto":
        """Repeat the data repeat_times times."""
        new_batch = None
        if self.batch is not None:
            new_batch = {}
            for key, tensor in self.batch.items():
                if interleave:
                    # [a, b] -> [a, a, b, b]
                    new_batch[key] = tensor.repeat_interleave(repeat_times, dim=0)
                else:
                    # [a, b] -> [a, b, a, b]
                    new_batch[key] = tensor.repeat(repeat_times, *([1] * (tensor.ndim - 1)))

        new_non_tensor_batch = None
        if self.non_tensor_batch is not None:
            new_non_tensor_batch = {}
            for key, arr in self.non_tensor_batch.items():
                if interleave:
                    # [a, b] -> [a, a, b, b]
                    new_non_tensor_batch[key] = np.repeat(arr, repeat_times)
                else:
                    # [a, b] -> [a, b, a, b]
                    new_non_tensor_batch[key] = np.tile(arr, repeat_times)

        return DataProto(
            batch=new_batch,
            non_tensor_batch=new_non_tensor_batch,
            meta_info=self.meta_info.copy() if self.meta_info else None,
        )

    def union(self, other: "DataProto") -> None:
        """Merge another DataProto into this one."""
        # Validate overlapping keys have the same values
        if self.batch is not None and other.batch is not None:
            for key in self.batch:
                if key in other.batch:
                    if not torch.all(self.batch[key] == other.batch[key]):
                        raise ValueError(f"Conflicting values for key '{key}'")

        # Merge batches
        if other.batch is not None:
            if self.batch is None:
                self.batch = {}
            for key, value in other.batch.items():
                if key not in self.batch:
                    self.batch[key] = value

        # Merge non-tensor batches
        if other.non_tensor_batch is not None:
            if self.non_tensor_batch is None:
                self.non_tensor_batch = {}
            for key, value in other.non_tensor_batch.items():
                if key not in self.non_tensor_batch:
                    self.non_tensor_batch[key] = value

        # Merge meta_info
        if other.meta_info is not None:
            if self.meta_info is None:
                self.meta_info = {}
            for key, value in other.meta_info.items():
                if key not in self.meta_info:
                    self.meta_info[key] = value


def pad_dataproto_to_divisor(data: DataProto, size_divisor: int) -> tuple:
    """Pad DataProto to make its length divisible by size_divisor."""
    length = len(data)
    remainder = length % size_divisor

    if remainder == 0:
        return data, 0

    pad_size = size_divisor - remainder

    # Pad by repeating the first elements
    pad_indices = list(range(pad_size))

    new_batch = None
    if data.batch is not None:
        new_batch = {}
        for key, tensor in data.batch.items():
            pad_tensor = tensor[pad_indices]
            new_batch[key] = torch.cat([tensor, pad_tensor], dim=0)

    new_non_tensor_batch = None
    if data.non_tensor_batch is not None:
        new_non_tensor_batch = {}
        for key, arr in data.non_tensor_batch.items():
            pad_arr = arr[pad_indices]
            new_non_tensor_batch[key] = np.concatenate([arr, pad_arr], axis=0)

    padded_data = DataProto(
        batch=new_batch,
        non_tensor_batch=new_non_tensor_batch,
        meta_info=data.meta_info.copy() if data.meta_info else None,
    )

    return padded_data, pad_size


def unpad_dataproto(data: DataProto, pad_size: int) -> DataProto:
    """Remove padding from DataProto."""
    if pad_size == 0:
        return data

    length = len(data)
    original_length = length - pad_size

    return data[:original_length]
