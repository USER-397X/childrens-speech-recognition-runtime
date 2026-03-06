import os
from pathlib import Path
from loguru import logger
from tqdm import tqdm

import torch
from omegaconf import OmegaConf, open_dict

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from nemo.core import adapter_mixins
from nemo.collections.speechlm2.models import SALM
from nemo.collections.speechlm2.data import SALMDataset, DataModule
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig

from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from lib.canary_qwen import CanaryQwenModel

#Attempted fix for OOM error
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Uncomment to save the model in local files
# model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
# model.save_pretrained('assets/canary-qwen')               

# Load Canary Qwen model
src_root = Path(__file__).parent.resolve()
model_dir = src_root / "assets" / "canary-qwen"
model = SALM.from_pretrained(model_dir)

# Log all parameter names and their shapes
for name, param in model.named_parameters():
    print(f"Parameter: {name} | Shape: {param.shape}")

# Log all module/layer names
for name, module in model.named_modules():
    print(f"Layer: {name}")

# 1. Define the adapter configuration
# Note: Your encoder's d_model is 1024, which the adapter mixin will fetch automatically,
# but you can define the bottleneck dimension (dim) here.
adapter_cfg = LinearAdapterConfig(
    in_features=1024, # Matches "d_model": 1024 from your config.json
    dim=64,           # The bottleneck dimension (you can tune this)
    norm_position="post",
    dropout=0.1
)

# 2. Add the adapter to the perception encoder
# This will iterate through all layers of the conformer and add the adapter.
model.perception.encoder.add_adapter(name="my_encoder_adapter", cfg=adapter_cfg)

# 3. Verify the adapter was added and enable it
if model.perception.encoder.is_adapter_available():
    model.perception.encoder.set_enabled_adapters(name="my_encoder_adapter", enabled=True)

# 4. Freeze the base model weights
# This ensures you only train the new adapter parameters, not the massive base model.
for param in model.parameters():
    param.requires_grad = False

# 5. Unfreeze ONLY the enabled adapters
# Iterate through the internal Conformer layers and unfreeze their specific adapters
for layer in model.perception.encoder.layers:
    layer.unfreeze_enabled_adapters()

# 1. Verify the adapter is registered and enabled
enabled_adapters = model.perception.encoder.get_enabled_adapters()
print(f"Enabled adapters: {enabled_adapters}") 
# Expected output: ['my_encoder_adapter'] (or whatever name you used)

# 2. Verify the base model is frozen and only the adapter is trainable
trainable_params = 0
frozen_params = 0

for name, param in model.perception.encoder.named_parameters():
    if param.requires_grad:
        print(f"Trainable parameter found: {name} | Shape: {param.shape}")
        trainable_params += param.numel()
    else:
        frozen_params += param.numel()

print(f"\nTrainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {frozen_params:,}")
# Expected output: Millions of frozen parameters, and a small fraction of trainable ones.


def train(cfg):
    # 1. Setup Checkpointing for your Adapters
    # This ensures we save the best versions of your model during training
    checkpoint_callback = ModelCheckpoint(
        dirpath="./canary_adapter_checkpoints",
        save_top_k=3,
        monitor="val_loss", # Make sure your specific NeMo model logs this metric
        mode="min",
        save_weights_only=True # We only care about saving the small adapter weights!
    )

def train():
    # 1. Setup Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="./canary_adapter_checkpoints",
        save_top_k=3,
        monitor="val_loss", 
        mode="min",
        save_weights_only=True 
    )

    # 2. Initialize the Trainer (using official precision settings)
    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=[1],               
        precision="16-mixed",   # Matched to official salm.yaml precision
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        accumulate_grad_batches=8,
    )

    # 3. Define Data Configs (Perfectly mirrored from official salm.yaml)
    train_data_config = {
        "sample_rate": 16000, 
        "batch_size": 1, 
        "max_duration": 40.0,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": True, 
        
        # ROOT-LEVEL parameters (Matched to salm.yaml)
        "prompt_format": model.cfg.prompt_format, 
        "token_equivalent_duration": 0.08,        
        
        "input_cfg": [               
            {
                "type": "lhotse_as_conversation", 
                "manifest_filepath": "data/train_small_manifest.jsonl", # Pointing to your manifest
                "audio_locator_tag": model.audio_locator_tag, 
                "tags": {
                    "context": "Transcribe the following:" 
                }
            }
        ]
    }

    val_data_config = {
        "sample_rate": 16000,
        "batch_size": 1,
        "shuffle": False, 
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": False, 
        
        # ROOT-LEVEL parameters (Matched to salm.yaml)
        "prompt_format": model.cfg.prompt_format,
        "token_equivalent_duration": 0.08,
        
        # NESTED Validation structure (Matched to salm.yaml)
        "datasets": {
            "my_val_set": {
                "input_cfg": [
                    {
                        "type": "lhotse_as_conversation",
                        "manifest_filepath": "data/val_small_manifest.jsonl",
                        "audio_locator_tag": model.audio_locator_tag,
                        "tags": {
                            "context": "Transcribe the following:"
                        }
                    }
                ]
            }
        }
    }

    # 4. Combine configs using OmegaConf
    data_config = OmegaConf.create({
        "train_ds": train_data_config,
        "validation_ds": val_data_config
    })

    # 5. Initialize Dataset and DataModule (Matched to official salm_train.py)
    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(data_config, tokenizer=model.tokenizer, dataset=dataset)

    # 6. Fit the model!
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()