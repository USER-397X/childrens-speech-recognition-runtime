from pathlib import Path
from loguru import logger
from tqdm import tqdm

import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

import nemo.collections.asr as nemo_asr

from nemo.collections.speechlm2.models import SALM
from lib.canary_qwen import CanaryQwenModel

# Uncomment to save the model in local files
# model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
# model.save_pretrained('assets/canary-qwen')               

# Load Canary Qwen model
src_root = Path(__file__).parent.resolve()
model_dir = src_root / "assets" / "canary-qwen"
model = SALM.from_pretrained(model_dir)

print(model.cfg)


def train():
    pass

if __name__ == "__main__":
    train()

