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

from typing import Any

from transformers import AutoProcessor, AutoTokenizer


def get_tokenizer(model_name: str, use_fast: bool = True) -> Any:
    """Load a tokenizer from HuggingFace.

    Args:
        model_name: The model name or path to load the tokenizer from.
        use_fast: Whether to use the fast tokenizer implementation.

    Returns:
        The loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, trust_remote_code=True)


def get_processor(model_name: str, use_fast: bool = True) -> Any:
    """Load a processor from HuggingFace (for vision-language models).

    Args:
        model_name: The model name or path to load the processor from.
        use_fast: Whether to use the fast tokenizer implementation.

    Returns:
        The loaded processor.
    """
    return AutoProcessor.from_pretrained(model_name, use_fast=use_fast, trust_remote_code=True)
