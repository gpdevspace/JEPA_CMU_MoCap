import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import requests

# 1. Configuration
REPO_ZIP_URL = "https://github.com/una-dinosauria/cmu-mocap/archive/refs/heads/master.zip"
TEMP_DIR = Path("data/raw/_temp_download")
OUTPUT_DIR = Path("data/raw")

# 2. The Tiered Curation Map
# Maps specific CMU MoCap Subject_Clip IDs to clear action categories.
# This ensures balanced classes and high-contrast motions for the latent space.
CURATED_ACTIONS = {
    "walking": ["16_15", "16_16", "16_31", "16_32", "35_01", "35_02", "35_03", "35_04", "07_01", "07_02"],
    "running": ["09_01", "09_02", "09_03", "09_04", "35_17", "35_18", "35_19", "35_20", "16_45", "16_46"],
    "jumping": ["13_11", "13_14", "16_11", "16_12", "02_04", "02_05", "16_13", "16_14"],
    "boxing":  ["13_17", "13_18", "14_01", "14_02", "14_03", "15_04", "15_05"],
    "dancing": ["05_01", "05_02", "05_03", "05_04", "05_05", "05_06", "05_07", "05_08", "05_09", "05_10"]
}

def download_and_extract():
    print(f"Downloading CMU MoCap archive (robust stream mode)...")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TEMP_DIR / "cmu_mocap.zip"
    
    # Use 'stream=True' to download in chunks
    with requests.get(REPO_ZIP_URL, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
    print("Download complete. Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)
    print("Extraction complete.")
    
def organize_files():
    print("Organizing curated actions...")
    # The zip creates a root folder named 'cmu-mocap-master'
    extracted_bvh_dir = Path("/Users/gpmac/Downloads/cmu-mocap-master/data")
    
    if not extracted_bvh_dir.exists():
        raise FileNotFoundError(f"Could not find the expected BVH directory at {extracted_bvh_dir}")

    total_moved = 0
    
    for action, clip_ids in CURATED_ACTIONS.items():
        action_dir = OUTPUT_DIR / action
        action_dir.mkdir(parents=True, exist_ok=True)
        
        for clip_id in clip_ids:
            # Subject ID is the part before the underscore (e.g., "16" from "16_15")
            subject_id = clip_id.split("_")[0]
            subject_id = "0" + subject_id
            
            source_file = extracted_bvh_dir / subject_id / f"{clip_id}.bvh"
            dest_file = action_dir / f"{clip_id}.bvh"
            
            if source_file.exists():
                shutil.copy2(source_file, dest_file)
                total_moved += 1
            else:
                print(f"Warning: Expected file {source_file} not found in archive.")
                
    print(f"Successfully organized {total_moved} files into {OUTPUT_DIR}.")

def cleanup():
    print("Cleaning up temporary download files...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    print("Done!")

if __name__ == "__main__":
    try:
        # download_and_extract()
        organize_files()
    finally:
        cleanup()