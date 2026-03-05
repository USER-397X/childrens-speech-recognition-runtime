from pathlib import Path
from loguru import logger
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict, OmegaConf



class ParakeetModel:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, model_path, train_path, val_path):
        logger.info(f"Loading model from: {model_path}")
        # model = nemo_asr.models.ASRModel.restore_from(model_path)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
        OmegaConf.set_struct(model.cfg.decoding, False)
        decoding_cfg = model.cfg.decoding
        decoding_cfg.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(decoding_cfg=decoding_cfg)
        OmegaConf.set_struct(model.cfg.decoding, True)

        return cls(model)

    def predict(self, audio_path: Path):
        hypotheses = self.model.transcribe([str(audio_path)], verbose=False)
        pred = hypotheses[0].text
        return pred

    def predict_batch(self, audio_paths: list[Path], batch_size: int = 4):
        hypotheses = self.model.transcribe(
            [str(p) for p in audio_paths], batch_size=batch_size, verbose=False
        )
        preds = [hyp.text for hyp in hypotheses]
        return preds