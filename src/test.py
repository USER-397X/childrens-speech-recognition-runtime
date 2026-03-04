from itertools import islice
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from loguru import logger
import torch
from tqdm import tqdm

from lib.parakeet import ParakeetModel
from metric.score import score_wer


BATCH_SIZE = 4
PROGRESS_STEP_DENOM = 100  # Update progress bar every 1 // PROGRESS_STEP_DENOM


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def main():
    # Diagnostics
    logger.info("Torch version: {}", torch.__version__)
    logger.info("CUDA available: {}", torch.cuda.is_available())
    logger.info("CUDA device count: {}", torch.cuda.device_count())

    # Load model
    src_root = Path(__file__).parent.resolve()
    model_path = src_root / "assets" / "parakeet-tdt-0.6b-v2" / "parakeet-tdt-0.6b-v2.nemo"
    logger.info(f"Loading model from: {model_path}")
    model = ParakeetModel.load(model_path)

    # Load manifest and process data
    data_dir = project_root / "data"
    manifest_path = data_dir / "val_manifest.jsonl"

    with manifest_path.open("r") as fr:
        items = [json.loads(line) for line in fr]

    # Sort by audio duration for better batching
    items.sort(key=lambda x: x["duration"], reverse=True)

    logger.info(f"Processing {len(items)} utterances from {manifest_path}")

    step = max(1, len(items) // PROGRESS_STEP_DENOM)

    # Predict
    actuals = []
    predictions = []
    next_log = step
    processed = 0
    logger.info("Starting transcription...")
    with open(os.devnull, "w") as devnull:
        with tqdm(total=len(items), file=devnull) as pbar:
            for batch in batched(items, BATCH_SIZE):
                preds = model.predict_batch(
                    [item["audio_filepath"] for item in batch],
                    batch_size=len(batch),
                )
                actuals.extend([item["text"] for item in batch])
                predictions.extend(preds)
                this_batch_size = len(batch)
                pbar.update(this_batch_size)
                processed += this_batch_size
                while processed >= next_log:
                    logger.info(str(pbar))
                    next_log += step

    logger.success("Transcription complete.")

    wer = score_wer(actuals, predictions)
    logger.info(f"Validation WER: {wer:.4f}")

    logger.success("Done.")


if __name__ == "__main__":
    main()