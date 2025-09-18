#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil

# allowed image extensions
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".bmp")

def normalize_sonar_images(input_dir: str, output_dir: str, gamma: float = None) -> None:
    """
    Load and normalize sonar images, preserving class subdirectories:
      - normalize to [0,1]
      - optional log or gamma transform
      - scale back to 8-bit
    Writes results into mirrored class subdirs under output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    # discover class subdirectories
    class_dirs = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]
    if not class_dirs:
        print(f"[WARN] No subdirectories found in '{input_dir}'. Nothing to do.")
        return

    # count all images across classes
    total_images = sum(
        len([f for f in os.listdir(os.path.join(input_dir, cls)) 
             if f.lower().endswith(IMAGE_EXTENSIONS)])
        for cls in class_dirs
    )
    print(f"[INFO] Found {total_images} image(s) across {len(class_dirs)} class folder(s) in '{input_dir}'")

    normalized_count = 0
    for cls in class_dirs:
        src_cls = os.path.join(input_dir, cls)
        dst_cls = os.path.join(output_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)

        for fname in os.listdir(src_cls):
            if not fname.lower().endswith(IMAGE_EXTENSIONS):
                continue
            img = cv2.imread(os.path.join(src_cls, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not read '{fname}', skipping.")
                continue

            # normalize to [0,1]
            img_f = img.astype(np.float32)
            img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8)

            # optional log or gamma
            if gamma is not None:
                if gamma == 0:
                    img_f = np.log1p(img_f)
                else:
                    img_f = np.power(img_f, gamma)
                img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8)

            # back to uint8 and write
            out_img = (img_f * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(dst_cls, fname), out_img)
            normalized_count += 1

    print(f"[INFO] Normalized and wrote {normalized_count} image(s) into '{output_dir}'")
    
    if is_flow:
        zip_base = "/workflow/outputs/normalized_images"
        shutil.make_archive(zip_base, 'zip', output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize sonar images (preserving class subdirs), local or in a Flow."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=f"/domino/datasets/local/{os.environ['DOMINO_PROJECT_NAME']}/unbalanced_training_validation_set",
        help="Local raw images dir; Flow jobs will instead read `/workflow/inputs/input_dir`"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"/domino/datasets/local/{os.environ['DOMINO_PROJECT_NAME']}/cleaned_data/normalized",
        help="Local output dir; Flow jobs will instead write to `/workflow/outputs/normalized_sonar_images`"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Optional gamma (0 for log, omit for no transform)"
    )
    args = parser.parse_args()

    # Determine if we’re running under a Flow
    is_flow = os.getenv("DOMINO_IS_WORKFLOW_JOB").lower() == "true"

    if is_flow:
        # Read the input directory 
        input_dir =  Path("/workflow/inputs/input_dir").read_text()       
        # Write outputs into the named output folder
        output_dir = "/tmp/normalized"
    
    else:
        input_dir = args.input_dir
        output_dir = args.output_dir

    normalize_sonar_images(input_dir, output_dir, args.gamma)
