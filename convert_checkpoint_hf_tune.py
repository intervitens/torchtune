
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
)

from torchtune.training.checkpointing._checkpointer import FullModelTorchTuneCheckpointer



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


checkpoint_dir = "../Qwen3-235B-A22B"
checkpoint_files = ["model.safetensors"]
output_dir = "./Qwen3-235B-A22B-tt"
model_type = "QWEN3_MOE"


_checkpoint_client_1 = CheckpointClient(cfg)
_checkpoint_client_2 = FullModelTorchTuneCheckpointer(checkpoint_dir, checkpoint_files, model_type, output_dir)
print("Loading HF ckpt")
checkpoint_dict = _checkpoint_client_1.load_base_checkpoint()
print("Saving torchtune ckpt")
_checkpoint_client_2.save_checkpoint(checkpoint_dict, 0, False)
print("Done!")
