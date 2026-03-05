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

from lib.canary_qwen import CanaryQwenModel

# Uncomment to save the model in local files
# model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
# model.save_pretrained('assets/canary-qwen')               

# Load Canary Qwen model
src_root = Path(__file__).parent.resolve()
model_dir = src_root / "assets" / "canary-qwen"
model = SALM.from_pretrained(model_dir)

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


def train():
    # 1. Setup Checkpointing for your Adapters
    # This ensures we save the best versions of your model during training
    checkpoint_callback = ModelCheckpoint(
        dirpath="./canary_adapter_checkpoints",
        save_top_k=3,
        monitor="val_loss", # Make sure your specific NeMo model logs this metric
        mode="min",
        save_weights_only=True # We only care about saving the small adapter weights!
    )

    # 2. Initialize the Trainer
    # We align the precision with the "bfloat16" torch_dtype from your config
    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,               # Set to >1 if you have a multi-GPU setup
        precision="bf16-mixed",  # Leverages the bfloat16 setup you configured
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        accumulate_grad_batches=4, # Helpful if your GPU memory is tight
    )

    train_data_config = {
        "sample_rate": 16000, 
        "batch_size": 8, 
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": True, 
        
        "use_lhotse": True,          
        "token_equivalent_duration": 0.08,        
        
        "input_cfg": [               
            {
                "type": "lhotse_as_conversation", 
                "manifest_filepath": "data/train_manifest.jsonl",
                "audio_locator_tag": model.audio_locator_tag, 
                "prompt_format": model.cfg.prompt_format,  # <--- MUST BE INSIDE THE LIST!
                "tags": {
                    "context": "Transcribe the following:" 
                }
            }
        ]
    }

    val_data_config = {
        "sample_rate": 16000,
        "batch_size": 8,
        "shuffle": False, 
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": True, 
        
        "use_lhotse": True,
        "token_equivalent_duration": 0.08,
        
        "input_cfg": [
            {
                "type": "lhotse_as_conversation",
                "manifest_filepath": "data/val_manifest.jsonl",
                "audio_locator_tag": model.audio_locator_tag,
                "prompt_format": model.cfg.prompt_format,  # <--- MUST BE INSIDE THE LIST!
                "tags": {
                    "context": "Transcribe the following:"
                }
            }
        ]
    }

    # 1. Combine your configs into a single OmegaConf dictionary
    data_config = OmegaConf.create({
        "train_ds": train_data_config,
        "validation_ds": val_data_config
    })

    # 2. Initialize the dataset and DataModule using the model's tokenizer
    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(data_config, tokenizer=model.tokenizer, dataset=dataset)

    # 3. Fit the model by passing BOTH the model and the datamodule to the trainer
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    train()