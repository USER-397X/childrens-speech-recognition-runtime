from pathlib import Path
from loguru import logger
import nemo.collections.asr as nemo_asr
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

class CanaryQwenModel:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls):
        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-qwen-2.5b")
        return cls(model)
    
    def train(self, train_manifest: Path, val_manifest: Path, epochs: int = 10, batch_size: int = 8):
        pass