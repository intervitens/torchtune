# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

import torch

from torchtune.models.convert_weights import get_mapped_key


# state dict key mappings from HF's format to torchtune's format
_FROM_HF = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attn.q_proj.bias",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attn.k_proj.bias",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attn.v_proj.bias",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attn.q_norm.scale",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attn.k_norm.scale",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
    "model.layers.{}.mlp.gate.weight": "layers.{}.mlp.router.gate.weight",
    "model.layers.{}.mlp.experts.0.gate_proj.weight": "layers.{}.mlp.experts.gate_proj",
    "model.layers.{}.mlp.experts.0.up_proj.weight": "layers.{}.mlp.experts.up_proj",
    "model.layers.{}.mlp.experts.0.down_proj.weight": "layers.{}.mlp.experts.down_proj",
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.mlp.shared_expert.w2.weight",
    "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.mlp.shared_expert.w1.weight",
    "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.mlp.shared_expert.w3.weight",
    "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.mlp.router.score_correction_bias",
    "model.norm.weight": "norm.scale",
    "lm_head.weight": "output.weight",
}


DOTS_TIED_KEY = "lm_head.weight"
DOTS_TUNE_EMBEDDING_KEY = "tok_embeddings.weight"


def get_mapped_key_moe(key: str, mapping_dict: dict[str, str]) -> str:
    """
    Maps a key from a model's state dictionary to a new key based on a mapping dictionary.

    This function is designed to handle keys that include layer numbers (e.g., "layer.0.attention").
    It correctly identifies the *first* number in the key as the layer index,
    replaces it with a placeholder '{}' to find the generic mapping, and then formats
    the new key with the original layer number. Other numbers in the key are left unchanged.

    Args:
        key (str): The original key from the state dictionary.
        mapping_dict (dict[str, str]): A dictionary mapping generic keys (with '{}' as a placeholder
                      for the layer number) to new generic keys.

    Returns:
        The newly mapped key.

    Raises:
        Exception: If the key cannot be found in the mapping dictionary, indicating a
                   mismatch in model formats.
    """
    try:
        match = re.search(r"\.(\d+)", key)

        if match:
            layer_num = match.group(1)
            abstract_key = f"{key[:match.start()]}.{{}}{key[match.end():]}"
            new_key = mapping_dict[abstract_key].format(layer_num)
        else:
            new_key = mapping_dict[key]

    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format."
        ) from e

    return new_key


def dots1_hf_to_tune(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    num_experts: int = 128,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to TorchTune's format, which contains the weights
    of a dots1 model.
    State dicts from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.
    The logic is identical to :func:`~torchtune.models.convert_weights.hf_to_tune`, but may not load
    output projection weights.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict in HF's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        num_experts (int): Number of experts in each MoE layer.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" in key:  # Skip loading the position embeddings
            continue
        if "experts.0" in key:
            new_key = get_mapped_key_moe(key, _FROM_HF)
            converted_state_dict[new_key] = torch.stack(
                [
                    state_dict[str(i).join(key.rsplit("0", 1))].T
                    for i in range(num_experts)
                ]
            )
        elif "experts" in key:
            continue
        else:
            new_key = get_mapped_key(key, _FROM_HF)
            converted_state_dict[new_key] = value
    return converted_state_dict


def dots1_tune_to_hf(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    num_experts: int = 128,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
):
    """
    Convert a state dict from torchtune's format to HF's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict in torchtune's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        num_experts (int): Number of experts in each MoE layer.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        dict[str, torch.Tensor]: State dict in HF's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}
    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        new_key = get_mapped_key_moe(key, inverted_mapping_dict)
        if "experts" in key:
            for i, tensor in enumerate(torch.unbind(value)):
                converted_state_dict[
                    str(i).join(new_key.rsplit("0", 1))
                ] = tensor.T.contiguous()
        else:
            converted_state_dict[new_key] = value

    return converted_state_dict
