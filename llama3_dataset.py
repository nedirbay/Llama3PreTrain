"""
Llama 3 Model Pre-training - Dataset Preparation
10 MB açyk çeşme iňlis dili dataseti taýýarlamak (FineWeb-Edu bilen täzelenen)
"""

import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
OUTPUT_FILE = RAW_DIR / "train.txt"
TARGET_SIZE_MB = 10  # Target size (MB)

def create_project_structure():
    """Create project structure"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/tokenizer",
        "models/checkpoints",
        "models/final",
        "logs",
        "outputs"
    ]
    
    print("[DIR] Creating project structure...")
    for d in dirs:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
    print("[OK] All directories ready.\n")

def prepare_real_dataset():
    """Load real datasets, merge and write to file"""
    
    print("[DOWN] Loading datasets...")

    # 1. WikiText-2 (Standard text)
    print("   -> Loading WikiText-2...")
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # 2. FineWeb-Edu (Modern open web text)
    # "sample-10BT" is a small subset of the massive dataset.
    print("   -> Loading FineWeb-Edu (Web content)...")
    web_data = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:2000]")

    # 3. TinyStories (Creative writing)
    print("   -> Loading TinyStories (Creative writing)...")
    stories = load_dataset("roneneldan/TinyStories", split="train[:2000]")

    print(f"\n[SAVE] Writing data to '{OUTPUT_FILE}'...")
    
    current_size = 0
    target_bytes = TARGET_SIZE_MB * 1024 * 1024
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Helper function to write datasets
        def write_dataset(dataset_obj, name):
            nonlocal current_size
            
            # Determine column name (text, content, etc.)
            text_column = "text"
            if "text" not in dataset_obj.features:
                if "content" in dataset_obj.features:  # Käbir datasetlerde 'content' bolýar
                    text_column = "content"
            
            count = 0
            for item in tqdm(dataset_obj, desc=f"Writing {name}"):
                text = item[text_column].strip()
                if text:
                    # Llama 3 pre-training formaty (EOS token bilen)
                    text = text + "\n\n<|endoftext|>\n\n"
                    f.write(text)
                    
                    # Ölçegi hasaplamak
                    current_size += len(text.encode('utf-8'))
                    count += 1
                    
                    if current_size >= target_bytes:
                        return True # Stop signal
            return False

        # 1. WikiText ýazmak
        if write_dataset(wikitext, "WikiText"):
            print("! Target size reached with WikiText.")
        
        # 2. Write FineWeb-Edu
        elif write_dataset(web_data, "FineWeb-Edu"):
            print("! Target size reached with FineWeb-Edu.")
            
        # 3. Write TinyStories
        elif write_dataset(stories, "TinyStories"):
            print("! Target size reached with TinyStories.")

    # Check result
    if os.path.exists(OUTPUT_FILE):
        final_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print("\n" + "=" * 50)
        print(f"[OK] Dataset successfully prepared!")
        print(f"[FILE] Location: {OUTPUT_FILE}")
        print(f"[STAT] Total size: {final_size:.2f} MB")
        print("=" * 50)
    else:
        print("[FAIL] File not created, an error occurred.")

if __name__ == "__main__":
    print("=" * 50)
    print("LLAMA 3 PRE-TRAINING - DATASET PREPARATION")
    print("=" * 50)
    
    create_project_structure()
    prepare_real_dataset()