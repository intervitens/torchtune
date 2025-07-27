from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
)

from torchtune.training.checkpointing._checkpointer import FullModelTorchTuneCheckpointer, FullModelHFCheckpointer
from omegaconf import DictConfig


checkpoint_dir_in = "./Qwen3-235B-A22B-Instruct-2507"
checkpoint_dir_out = "./Qwen3-235B-A22B-tt-ckpt"
checkpoint_files_out = ["model-00001-of-00001.bin"]
output_dir = "./Qwen3-235B-A22B-Instruct-2507-tt-ckpt"
model_type = "QWEN3_MOE"

checkpoint_files_in = {
    "filename_format": "model-{}-of-{}.safetensors",
    "max_filename": "00118",
}

_checkpoint_client_1 = FullModelHFCheckpointer(checkpoint_dir_in, checkpoint_files_in, model_type, output_dir)
_checkpoint_client_2 = FullModelTorchTuneCheckpointer(checkpoint_dir_out, checkpoint_files_out, model_type, output_dir)

print("Loading HF ckpt")
checkpoint_dict = _checkpoint_client_1.load_checkpoint()
print("Saving torchtune ckpt")
_checkpoint_client_2.save_checkpoint(checkpoint_dict, 0, False)
print("Done!")
