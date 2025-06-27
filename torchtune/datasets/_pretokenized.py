# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Union

import torch
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
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.prepacked = (
            "input_pos" in self._data.column_names
            and "seq_lens" in self._data.column_names
        )
        if ("input_pos" in self._data.column_names) != (
            "seq_lens" in self._data.column_names
        ):
            raise ValueError(
                'Pre-packed PretokenizedDataset requires both "input_pos" and "seq_lens" columns to be present in the dataset.'
            )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        if self.prepacked:
            return {
                "tokens": torch.tensor(self._data[index]["tokens"], dtype=torch.long),
                "labels": torch.tensor(self._data[index]["labels"], dtype=torch.long),
                "input_pos": torch.tensor(
                    self._data[index]["input_pos"], dtype=torch.long
                ),
                "seq_lens": torch.tensor(
                    self._data[index]["seq_lens"], dtype=torch.long
                ),
            }
        else:
            sample = self._data[index]
            return sample


def pretokenized_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    packed: bool = False,
    split_across_pack: bool = False,
    **load_dataset_kwargs: dict[str, Any],
) -> Union[PretokenizedDataset, PackedDataset]:
    ds = PretokenizedDataset(
        tokenizer=tokenizer,
        source=source,
        **load_dataset_kwargs,
    )
    if packed and not ds.prepacked:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(
            ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=split_across_pack
        )
    return ds
