import json
import random
from pathlib import Path

def create_nemo_manifests(
    input_manifest: Path, 
    data_dir: Path, 
    train_output: Path, 
    val_output: Path, 
    train_split: float = 0.9
):
    print(f"Reading from {input_manifest}...")
    
    with open(input_manifest, 'r') as f:
        lines = f.readlines()

    nemo_items = []
    
    for line in lines:
        item = json.loads(line)
        
        # 1. Resolve the absolute path to the audio file
        # NeMo trainers work best with absolute paths so it doesn't matter 
        # where you run the training script from.
        abs_audio_path = (data_dir / item["audio_path"]).resolve()
        
        # 2. Map your keys to NeMo's required keys
        nemo_item = {
            "audio_filepath": str(abs_audio_path),
            "duration": item["audio_duration_sec"],
            "text": item["orthographic_text"]
        }
        nemo_items.append(nemo_item)

    # 3. Shuffle the data before splitting
    print("Shuffling and splitting data...")
    random.seed(42) # For reproducibility
    random.shuffle(nemo_items)

    # 4. Split into Train and Validation sets
    split_idx = int(len(nemo_items) * train_split)
    train_items = nemo_items[:split_idx]
    val_items = nemo_items[split_idx:]

    # 5. Write the Train Manifest
    with open(train_output, 'w') as fw:
        for item in train_items:
            fw.write(json.dumps(item) + "\n")

    # 6. Write the Validation Manifest
    with open(val_output, 'w') as fw:
        for item in val_items:
            fw.write(json.dumps(item) + "\n")

    print("Done!")
    print(f" - Total utterances: {len(nemo_items)}")
    print(f" - Training items:   {len(train_items)} -> saved to {train_output}")
    print(f" - Validation items: {len(val_items)} -> saved to {val_output}")


if __name__ == "__main__":
    # Define paths based on your project structure
    # Assuming this script is run from the project root or src directory
    data_directory = Path("data").resolve()
    
    input_metadata = data_directory / "utterance_metadata.jsonl"
    train_manifest = data_directory / "train_nemo_manifest.jsonl"
    val_manifest = data_directory / "val_nemo_manifest.jsonl"
    
    if not input_metadata.exists():
        print(f"Error: Could not find {input_metadata}")
    else:
        create_nemo_manifests(
            input_manifest=input_metadata,
            data_dir=data_directory,
            train_output=train_manifest,
            val_output=val_manifest,
            train_split=0.9
        )