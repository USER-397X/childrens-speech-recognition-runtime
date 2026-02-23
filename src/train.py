from pathlib import Path
from lib.parakeet import ParakeetModel

def main():
    src_root = Path(__file__).parent.resolve()
    model_path = src_root / "assets" / "parakeet-tdt-0.6b-v2" / "parakeet-tdt-0.6b-v2.nemo"
    
    # 1. Load the base model
    model = ParakeetModel.load(model_path)
    
    # 2. Define manifest paths
    # Note: See Step 3 below about formatting these files!
    data_dir = Path("data")
    train_manifest = data_dir / "train_nemo_manifest.jsonl"
    val_manifest = data_dir / "val_nemo_manifest.jsonl"
    
    # 3. Start training
    model.train(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        epochs=3,
        batch_size=2
    )

if __name__ == "__main__":
    main()