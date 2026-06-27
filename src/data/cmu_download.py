import os
import shutil
from pathlib import Path

# 1. Configuration - Pointing directly to your local manual download
EXTRACTED_ZIP_DIR = Path("/Users/gpmac/Downloads/cmu-mocap-master")
OUTPUT_DIR = Path("data/raw")
TEMP_DIR = Path("data/raw/_temp_download")

# 2. Fully Expanded 3-Class Curation Map (30 Clips per class = 90 Total clips)
# Drastically scaled up with high-quality sequences from subjects 02, 05, 07, 08, 09, 10, 12, 13, 14, 15, 16, 26, 35, and 49.
CURATED_ACTIONS = {
    "walking": [
        "16_15", "16_16", "16_31", "16_32", "35_01", "35_02", "35_03", "35_04", 
        "07_01", "07_02", "08_01", "08_02", "10_04", "12_01", "12_02", "02_01", 
        "02_02", "05_01", "07_03", "07_04", "08_03", "08_04", "08_05", "08_06",
        "16_21", "16_22", "35_05", "35_06", "35_07", "35_08"
    ],
    "jumping": [
        "13_11", "13_14", "16_11", "16_12", "02_04", "02_05", "16_13", "16_14",
        "49_01", "49_02", "49_03", "49_04", "49_05", "02_03", "13_10", "13_12", 
        "13_13", "13_15", "16_01", "16_02", "16_03", "16_04", "16_05", "16_06",
        "16_07", "16_08", "49_06", "49_07", "49_08", "49_09"
    ],
    "boxing":  [
        "13_17", "13_18", "14_01", "14_02", "14_03", "15_04", "15_05",
        "13_19", "13_20", "14_04", "26_01", "26_02", "13_21", "13_22", "13_23", 
        "13_24", "14_05", "14_06", "14_07", "14_08", "14_09", "14_10", "14_11", 
        "15_01", "15_02", "15_03", "15_06", "26_03", "26_04", "26_05"
    ]
}

def organize_files():
    print("Organizing expanded curated actions...")
    
    # Keeping your custom data sub-folder layout intact
    extracted_bvh_dir = EXTRACTED_ZIP_DIR / "data"
    
    if not extracted_bvh_dir.exists():
        raise FileNotFoundError(f"Could not find the expected BVH directory at {extracted_bvh_dir}")

    # Completely flush output folder before writing to avoid mixing old/new classes
    if OUTPUT_DIR.exists():
        print(f"Clearing old raw contents from {OUTPUT_DIR}...")
        for child in OUTPUT_DIR.iterdir():
            if child.is_dir() and child.name in ["walking", "running", "jumping", "boxing", "dancing", "other"]:
                shutil.rmtree(child)

    total_moved = 0
    total_expected = sum(len(clips) for clips in CURATED_ACTIONS.values())
    
    for action, clip_ids in CURATED_ACTIONS.items():
        action_dir = OUTPUT_DIR / action
        action_dir.mkdir(parents=True, exist_ok=True)
        
        for clip_id in clip_ids:
            subject_id = clip_id.split("_")[0]
            # Keeping your prefix handling exactly as you observed it
            subject_id = "0" + subject_id
            
            source_file = extracted_bvh_dir / subject_id / f"{clip_id}.bvh"
            dest_file = action_dir / f"{clip_id}.bvh"
            
            if source_file.exists():
                shutil.copy2(source_file, dest_file)
                total_moved += 1
            else:
                print(f"Warning: Expected file {source_file} not found in archive.")
                
    print(f"Successfully organized {total_moved}/{total_expected} files into {OUTPUT_DIR}.")

def cleanup():
    print("Cleaning up temporary download folders (if any)...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    print("Done!")

if __name__ == "__main__":
    try:
        organize_files()
    finally:
        cleanup()