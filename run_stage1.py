import os
import subprocess
import sys
 
LESION_TYPES = ["ex", "he", "ma"]
 
def patch_config(target_lesion):
    config_path = "segmentation/config.py"
 
    with open(config_path, "r") as f:
        lines = f.readlines()
 
    # rewrite in-place to avoid stale module cache between subprocess calls
    with open(config_path, "w") as f:
        for line in lines:
            if line.strip().startswith("TARGET_LESION"):
                f.write(f'    TARGET_LESION = "{target_lesion}"\n')
            else:
                f.write(line)
 
def already_trained(target_lesion):
    # skip lesions that completed in a previous session
    checkpoint_path = os.path.join("/kaggle/working", f"best_{target_lesion}_model.pth")
    return os.path.exists(checkpoint_path)
 
def run_training():
    result = subprocess.run(
        [sys.executable, "segmentation/train_monai.py"],
        check=False,
    )
    return result.returncode
 
if __name__ == "__main__":
    for lesion in LESION_TYPES:
        if already_trained(lesion):
            print(f"\nSkipping {lesion.upper()} — checkpoint already exists in /kaggle/working.")
            continue
 
        print(f"\nStarting Stage 1 Training for: {lesion.upper()}...")
 
        patch_config(lesion)
        exit_code = run_training()
 
        # abort early so a broken run doesn't silently waste GPU time on the next lesion
        if exit_code != 0:
            print(f"Training for {lesion.upper()} failed with exit code {exit_code}. Aborting.")
            sys.exit(exit_code)
 
    print("All specialist models trained successfully.")
