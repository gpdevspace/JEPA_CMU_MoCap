import json
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    print("="*60)
    print("         CRITICAL DATA PIPELINE SANITY AUDIT (RAW)")
    print("="*60)

    # 1. Paths
    root_dir = Path("/Users/gpmac/gpbuildspace/JEPA/JEPA_CMU_MoCap")
    processed_dir = root_dir / "data" / "processed" # Adjust if your config path differs
    
    if not processed_dir.exists():
        # Fallback check
        processed_dir = root_dir / "outputs" / "processed"
        if not processed_dir.exists():
            print(f"❌ ERROR: Cannot find processed directory. Check your paths.")
            return

    skeleton_path = processed_dir / "skeleton.json"
    if not skeleton_path.exists():
        print(f"❌ ERROR: Cannot find skeleton.json at {skeleton_path}")
        return
        
    with open(skeleton_path) as f:
        meta = json.load(f)
    bone_pairs = meta.get("bone_pairs", [])
    print(f"✅ Metadata Loaded. Defined Bones Count: {len(bone_pairs)}")

    # 2. Find a raw processed numpy file
    npz_files = list(processed_dir.glob("*.npz"))
    if not npz_files:
        print(f"❌ ERROR: No compiled .npz files found in {processed_dir}")
        return
    
    target_file = npz_files[0]
    print(f"✅ Loading raw data file: {target_file.name}")
    data = np.load(target_file)
    
    # Assuming standard keys like 'poses' or 'data'
    poses_key = 'poses' if 'poses' in data else list(data.keys())[0]
    raw_poses = data[poses_key] # shape usually [frames, dimensions]
    
    print(f"✅ Array loaded. Shape: {raw_poses.shape}")
    
    if len(raw_poses) < 2:
        print("❌ ERROR: File contains fewer than 2 frames. Cannot compute deltas.")
        return

    pose_t = raw_poses[0]
    pose_t1 = raw_poses[1]
    
    # Check shapes
    num_coordinates = pose_t.shape[0]
    num_joints = num_coordinates // 3
    print(f"✅ Active Features per frame: {num_coordinates} (Inferred Joints: {num_joints})")
    
    pose_3d = pose_t.reshape(num_joints, 3)
    deltas = (pose_t1 - pose_t).reshape(num_joints, 3)
    delta_magnitudes = np.linalg.norm(deltas, axis=1)

    print("\n" + "-"*40)
    print(" STATISTICAL HEALTH CHECKS")
    print("-"*40)
    print(f"Pose coordinate Range: Min = {pose_t.min():.3f}, Max = {pose_t.max():.3f}")
    print(f"Frame-to-Frame Delta:  Min = {deltas.min():.5f}, Max = {deltas.max():.5f}")
    print(f"Average Mean Joint Velocity per frame: {delta_magnitudes.mean():.5f}")
    
    has_nan = np.isnan(pose_t).any() or np.isnan(pose_t1).any()
    has_zero_var = np.std(pose_t) < 1e-4
    
    print(f"Any NaNs in data?      {'❌ YES (CRITICAL BUG)' if has_nan else '✅ No'}")
    print(f"Dead/Static Channels?  {'❌ YES (CRITICAL BUG)' if has_zero_var else '✅ No'}")

    # 3. TOPOLOGY & BONE RIGIDITY TEST
    print("\n" + "-"*40)
    print(" TOPOLOGY & BONE RIGIDITY TEST")
    print("-"*40)
    
    broken_bones = 0
    max_joint_idx_found = 0
    
    for idx, (parent, child) in enumerate(bone_pairs):
        max_joint_idx_found = max(max_joint_idx_found, parent, child)
        if parent >= num_joints or child >= num_joints:
            broken_bones += 1
            continue
            
        len_t = np.linalg.norm(pose_3d[parent] - pose_3d[child])
        len_t1 = np.linalg.norm((pose_t1.reshape(num_joints, 3))[parent] - (pose_t1.reshape(num_joints, 3))[child])
        
        if idx < 3:
            print(f"Bone {idx} (Joint {parent}->{child}): Len(t)={len_t:.3f}, Len(t+1)={len_t1:.3f} | Change={abs(len_t - len_t1):.5f}")

    print(f"\nMax joint index referenced in bone_pairs: {max_joint_idx_found}")
    if max_joint_idx_found >= num_joints:
        print(f"❌ CRITICAL TOPOLOGY MISMATCH: Bone metadata references joint index {max_joint_idx_found}, but array data only has {num_joints} joints!")
    if broken_bones > 0:
        print(f"❌ CRITICAL METADATA BUG: {broken_bones} bones point to joint indices out of bounds.")
    elif max_joint_idx_found < num_joints - 1:
        print(f"⚠️ WARNING: Some joints in your data are completely disconnected orphans (not included in bone_pairs).")
    else:
        print("✅ SUCCESS: Bone metadata indexing boundaries match the data array dimensions.")

    # 4. Visual Check
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pose_3d[:, 0], pose_3d[:, 2], pose_3d[:, 1], c='crimson', s=20)
    
    for parent, child in bone_pairs:
        if parent < num_joints and child < num_joints:
            p = pose_3d[parent]
            c = pose_3d[child]
            ax.plot([p[0], c[0]], [p[2], c[2]], [p[1], c[1]], color='blue')
            
    ax.set_title("Ground Truth Raw Frame Rig Check")
    output_img = root_dir / "outputs" / "data_sanity_check.png"
    output_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img)
    print(f"\n📸 Visual verification frame saved to: {output_img}")

if __name__ == "__main__":
    main()