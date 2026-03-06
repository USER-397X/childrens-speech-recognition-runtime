import os
import torch
from pathlib import Path
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

adapter_cfg = LinearAdapterConfig(
    in_features=1024, # Matches "d_model": 1024 from your config.json
    dim=64,           # The bottleneck dimension (you can tune this)
    norm_position="post",
    dropout=0.1
)

@hydra_runner(config_path="conf", config_name="salm")
def train(cfg):
    OmegaConf.resolve(cfg)
    
    torch.set_float32_matmul_precision("medium")
    
    # 2. Update Trainer config for single GPU
    cfg.trainer.devices = [1]  # Targets GPU 1
    cfg.trainer.accelerator = "gpu"
    cfg.trainer.strategy = "auto" # "ddp" is for multi-gpu; "auto" works for 1
    
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        src_root = Path(__file__).parent.resolve()
        model_dir = src_root / "assets" / "canary-qwen"
        model = SALM.from_pretrained(model_dir)

        model.perception.encoder.add_adapter(name="my_encoder_adapter", cfg=adapter_cfg)

        if model.perception.encoder.is_adapter_available():
            model.perception.encoder.set_enabled_adapters(name="my_encoder_adapter", enabled=True)

        for param in model.parameters():
            param.requires_grad = False

        for layer in model.perception.encoder.layers:
            layer.unfreeze_enabled_adapters()

    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train()