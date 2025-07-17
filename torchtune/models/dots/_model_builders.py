# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional


from torchtune.models.dots1._component_builders import lora_dots1, dots1
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES

"""
Model builders build specific instantiations using component builders. For example
the qwen3_8b_instruct model builder uses the qwen2 component builder to create the
Qwen3 8B instruct model.
"""


def dots1_143B_A14B() -> TransformerDecoder:
    """
    Builder for creating a dots1 model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-235B-A22B

    Returns:
        TransformerDecoder: Instantiation of dots1 model
    """
    return dots1(
        vocab_size=152064,
        num_layers=62,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        intermediate_dim=10944,
        moe_intermediate_size=1408,
        first_k_dense_replace=1,
        num_experts=128,
        num_experts_per_tok=6,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10000000.0,
        routed_scaling_factor=2.5,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )

def dots1_test_16b() -> TransformerDecoder:
    """
    Builder for creating a dots1 model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-235B-A22B

    Returns:
        TransformerDecoder: Instantiation of dots1 model
    """
    return dots1(
        vocab_size=152064,
        num_layers=27,
        num_heads=16,
        num_kv_heads=16,
        embed_dim=2048,
        intermediate_dim=10944,
        moe_intermediate_size=1408,
        first_k_dense_replace=1,
        num_experts=64,
        num_shared_experts=2,
        num_experts_per_tok=6,
        max_seq_len=8192,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10000000.0,
        routed_scaling_factor=1.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def lora_dots1_143B_A14B(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a dots1 base model with LoRA enabled.

    The dots1 defaults are the same as in :func:`~torchtune.models.dots1.dots1_143B_A14B`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of dots1 model with LoRA applied
    """
    return lora_dots1(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=152064,
        num_layers=27,
        num_heads=16,
        num_kv_heads=16,
        embed_dim=2048,
        intermediate_dim=10944,
        first_k_dense_replace=1,
        moe_intermediate_size=1408,
        num_experts=64,
        num_shared_experts=2,
        num_experts_per_tok=6,
        max_seq_len=8192,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10000000.0,
        routed_scaling_factor=2.5,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )

def lora_dots1_test_16b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a dots1 base model with LoRA enabled.

    The dots1 defaults are the same as in :func:`~torchtune.models.dots1.dots1_143B_A14B`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of dots1 model with LoRA applied
    """
    return lora_dots1(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=152064,
        num_layers=27,
        num_heads=16,
        num_kv_heads=16,
        embed_dim=2048,
        intermediate_dim=10944,
        moe_intermediate_size=1408,
        first_k_dense_replace=1,
        num_experts=64,
        num_shared_experts=2,
        num_experts_per_tok=6,
        max_seq_len=8192,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10000000.0,
        routed_scaling_factor=1.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )
