import os
import sys
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from metric.score import score_wer
from lib.parakeet import ParakeetModel

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def main():
    logger.info("Starting validation evaluation")
    src_root = project_root / "src"
    model_dir = src_root / "assets" / "parakeet-tdt-0.6b-v2" / "parakeet-tdt-0.6b-v2.nemo"
    
    logger.info(f"Loading Parakeet offline from {model_dir}...")
    model = ParakeetModel.load(model_path=model_dir)

    data_dir = project_root / "data"
    val_manifest = data_dir / "val_manifest.jsonl"
    
    items = []
    with open(val_manifest, "r") as f:
        for line in f:
            items.append(json.loads(line))
            
    # Subsample for testing if testing arg is passed
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        items = items[:16]
            
    # Sort by duration for better batching
    items.sort(key=lambda x: x.get("duration", 0), reverse=True)
    
    batch_size = 16
    actuals = []
    predicteds = []
    
    logger.info(f"Evaluating {len(items)} validation samples...")
    for batch in tqdm(list(batched(items, batch_size))):
        audio_paths = [Path(item["audio_filepath"]) for item in batch]
        texts = [item["text"] for item in batch]
        
        preds = model.predict_batch(audio_paths, batch_size=len(batch))
        
        actuals.extend(texts)
        predicteds.extend(preds)
        
    try:
        wer = score_wer(actuals, predicteds)
        logger.info(f"Validation WER: {wer:.4f}")
    except Exception as e:
        logger.error(f"Failed to calculate WER: {e}")
        debug_file = project_root / "eval_val_debug.json"
        with open(debug_file, "w") as f:
            json.dump({"actuals": actuals, "predicteds": predicteds}, f, indent=2)
        logger.info(f"Saved actuals and predicteds to {debug_file}")

if __name__ == "__main__":
    main()
