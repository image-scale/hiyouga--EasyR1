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

    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        non_tensors: Optional[Dict[str, np.ndarray]] = None,
        meta_info: Optional[Dict[str, Any]] = None,
        num_batch_dims: int = 1,
    ) -> "DataProto":
        raise NotImplementedError

    @classmethod
    def from_single_dict(cls, data: Dict[str, Any]) -> "DataProto":
        raise NotImplementedError

    @classmethod
    def concat(cls, data_list: List["DataProto"]) -> "DataProto":
        raise NotImplementedError

    @classmethod
    def load_from_disk(cls, path: str) -> "DataProto":
        raise NotImplementedError

    def save_to_disk(self, path: str) -> None:
        raise NotImplementedError

    def select(
        self,
        batch_keys: Optional[List[str]] = None,
        non_tensor_batch_keys: Optional[List[str]] = None,
        meta_info_keys: Optional[List[str]] = None,
    ) -> "DataProto":
        raise NotImplementedError

    def pop(
        self,
        batch_keys: Optional[List[str]] = None,
        non_tensor_batch_keys: Optional[List[str]] = None,
        meta_info_keys: Optional[List[str]] = None,
    ) -> "DataProto":
        raise NotImplementedError

    def chunk(self, chunks: int) -> List["DataProto"]:
        raise NotImplementedError

    def split(self, split_size: int) -> List["DataProto"]:
        raise NotImplementedError

    def reorder(self, indices: torch.Tensor) -> None:
        raise NotImplementedError

    def repeat(self, repeat_times: int, interleave: bool = False) -> "DataProto":
        raise NotImplementedError

    def union(self, other: "DataProto") -> None:
        raise NotImplementedError


def pad_dataproto_to_divisor(data: DataProto, size_divisor: int) -> tuple:
    raise NotImplementedError


def unpad_dataproto(data: DataProto, pad_size: int) -> DataProto:
    raise NotImplementedError
