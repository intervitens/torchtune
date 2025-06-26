from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
)

from torchtune.training.checkpointing._checkpointer import FullModelTorchTuneCheckpointer, FullModelHFCheckpointer
from omegaconf import DictConfig

cfg = {
    "device": "cpu",
    "checkpointer": {
        "_component_": "torchtune.training.FullModelHFCheckpointer",
        "checkpoint_dir": "../Qwen3-235B-A22B",
        "checkpoint_files": {
            "filename_format": "model-{}-of-{}.safetensors",
            "max_filename": "00118",
        },
        "recipe_checkpoint": None,
        "output_dir": "./output",
        "model_type": "QWEN3_MOE",
    },
}

cfg = DictConfig(cfg)


checkpoint_dir_in = "./qwen3_235B_A22B_hydrus/full/epoch_1"
checkpoint_dir_out = "./Qwen3-235B-A22B"
checkpoint_files_in = ["model-00001-of-00001.bin"]
output_dir = "./Qwen3-235B-A22B-hydrus-HF"
model_type = "QWEN3_MOE"

checkpoint_files_out = {
    "filename_format": "model-{}-of-{}.safetensors",
    "max_filename": "00118",
}

_checkpoint_client_1 = FullModelTorchTuneCheckpointer(checkpoint_dir_in, checkpoint_files_in, model_type, output_dir)
_checkpoint_client_2 = FullModelHFCheckpointer(checkpoint_dir_out, checkpoint_files_out, model_type, output_dir)
print("Loading torchtune ckpt")
checkpoint_dict = _checkpoint_client_1.load_checkpoint()
print("Saving HF ckpt")
_checkpoint_client_2.save_checkpoint(checkpoint_dict, 0, False)
print("Done!")