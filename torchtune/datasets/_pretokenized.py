# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Union

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


class PretokenizedDataset(Dataset):
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        packed: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return sample


def pretokenized_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    packed: bool = False,
    split_across_pack: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[PretokenizedDataset, PackedDataset]:
    ds = PretokenizedDataset(
        tokenizer=tokenizer,
        source=source,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(
            ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=split_across_pack
        )
    return ds
