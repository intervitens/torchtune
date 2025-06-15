import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from torchtune.models.qwen3 import (
    qwen3_8b_base,
    qwen3_moe_30b_a3b_base,
    qwen3_moe_15b_a2b_base,
    qwen3_tokenizer,
)
from torchtune.models.llama3 import llama3_8b, llama3_tokenizer
from torchtune.models.qwen3._convert_weights import qwen3_hf_to_tune, qwen3_moe_hf_to_tune
from torchtune import config, generation, training, utils
from torchtune.models.convert_weights import hf_to_tune

def llama_hf_to_tune(hf_model):
    return hf_to_tune(
            hf_model.state_dict(),
            num_heads=hf_model.config.num_attention_heads,
            num_kv_heads=hf_model.config.num_key_value_heads,
            dim=hf_model.config.hidden_size,
            head_dim=hf_model.config.head_dim,
        )

MODELS = {
    # "qwen": {
    #     "variants": {
    #         "8b": (qwen3_8b_base, "./Qwen3-8B-Base"),
    #     },
    #     "tokenizer": qwen3_tokenizer,
    #     "convert": qwen3_hf_to_tune,
    # },
    "qwen_moe": {
       "variants": {
            "30b": (qwen3_moe_30b_a3b_base, "../Qwen3-30B-A3B-Base"),
            #"15b": (qwen3_moe_15b_a2b_base, "../Qwen3-15B-A2B-Base"),
       },
       "tokenizer": qwen3_tokenizer,
       "convert": qwen3_moe_hf_to_tune,
    },
    # "llama": {
    #     "variants": {
    #         "8b": (llama3_8b, "../Meta-Llama-3-8B"),
    #     },
    #     "tokenizer": llama3_tokenizer,
    #     "convert": llama_hf_to_tune,
    # },
}

def compare_logits(model_name, variant_key, builder, hf_name, convert_fn):
    print(f"Comparing {model_name} {variant_key}")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_name)

    dtype = torch.bfloat16
    print("Using dtype", dtype)
    _device = utils.get_device(device="cuda")
    with training.set_default_dtype(dtype), _device:
    #with _device:
        # Create tune model
        tune_model = builder()
    
    if convert_fn is not None:
        if model_name == "llama":
            converted_sd = convert_fn(hf_model)
        else:
            converted_sd = convert_fn(hf_model.state_dict())
        tune_model.load_state_dict(converted_sd, strict=True)

    # Use the HF tokenizer for simplicity
    input_str = "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive"
    #input_str = "Hello world"
    tokens = hf_tokenizer(input_str, return_tensors="pt").input_ids
    tokens_cuda = tokens.to(_device)
    tokens_cuda_1 = tokens.to("cuda:1")
    with torch.no_grad():
        tune_logits = tune_model(tokens_cuda).to("cpu")
        hf_model = hf_model.to("cuda:1", dtype)
        hf_logits = hf_model(tokens_cuda_1).logits.to("cpu")
        
    diff = (hf_logits - tune_logits).abs().max().item()
    print(f"Max logit diff: {diff:.6f}\n")


def main():
    for name, info in MODELS.items():
        for variant, (builder, hf_name) in info["variants"].items():
            compare_logits(
                name,
                variant,
                builder,
                hf_name,
                info["convert"],
            )


if __name__ == "__main__":
    main()  
