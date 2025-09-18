#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
import shutil

# allowed image extensions
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".bmp")

def unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    sharp = img.astype(np.float32) + amount * mask
    if threshold > 0:
        low_contrast = np.abs(img.astype(np.float32) - blurred.astype(np.float32)) < threshold
        sharp[low_contrast] = img[low_contrast]
    return np.clip(sharp, 0, 255).astype(np.uint8)

def enhance_features(input_dir: str,
                     output_dir: str,
                     clahe_clip: float,
                     clahe_grid: int,
                     unsharp_sigma: float,
                     unsharp_amount: float,
                     unsharp_threshold: int):
    """
    Enhance contrast & details on sonar images, preserving class subdirs:
      - CLAHE
      - Unsharp mask
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
    total_images = 0
    for cls in class_dirs:
        cls_path = os.path.join(input_dir, cls)
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
        total_images += len(imgs)
    print(f"[INFO] Found {total_images} image(s) across {len(class_dirs)} class folder(s) in '{input_dir}'")

    processed = 0
    clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                            tileGridSize=(clahe_grid, clahe_grid))

    for cls in class_dirs:
        src_cls = os.path.join(input_dir, cls)
        dst_cls = os.path.join(output_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)

        for fname in os.listdir(src_cls):
            if not fname.lower().endswith(IMAGE_EXTENSIONS):
                continue
            src_path = os.path.join(src_cls, fname)
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not read '{src_path}', skipping.")
                continue

            # CLAHE
            cl = clahe.apply(img)
            # Unsharp mask
            final = unsharp_mask(cl,
                                  sigma=unsharp_sigma,
                                  amount=unsharp_amount,
                                  threshold=unsharp_threshold)

            dst_path = os.path.join(dst_cls, fname)
            cv2.imwrite(dst_path, final)
            processed += 1

    print(f"[INFO] Enhanced and wrote {processed} image(s) into '{output_dir}'")
    
    if is_flow:
        # Write outputs into the named output folder
        zip_base = "/workflow/outputs/enhanced_images"
        shutil.make_archive(zip_base, 'zip', output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhance sonar images (preserving class subdirs), local or in a Flow."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=f"/domino/datasets/local/{os.environ['DOMINO_PROJECT_NAME']}/cleaned_data/noise_removal",
        help="Local input (step2 output); Flow jobs read `/workflow/inputs/input_dir`"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"/domino/datasets/local/{os.environ['DOMINO_PROJECT_NAME']}/cleaned_data/final",
        help="Local output dir; Flow jobs write to `/workflow/outputs/enhanced_sonar_images`"
    )
    parser.add_argument("--clahe_clip",    type=float, default=2.0)
    parser.add_argument("--clahe_grid",    type=int,   default=8)
    parser.add_argument("--unsharp_sigma", type=float, default=1.0)
    parser.add_argument("--unsharp_amount",type=float, default=1.5)
    parser.add_argument("--unsharp_threshold", type=int, default=0)
    args = parser.parse_args()

    # Detect Flow mode
    is_flow = os.getenv("DOMINO_IS_WORKFLOW_JOB", "false").lower() == "true"

    if is_flow:
        # Flow‐mounted input & output paths
        input_zip = "/workflow/inputs/denoised_images"
        tmp_in = "/tmp/denoised"
        shutil.unpack_archive(input_zip, tmp_in, 'zip')
        input_dir = tmp_in
        output_dir = "/tmp/enhanced"
    
    else:
        # Local development
        input_dir  = args.input_dir
        output_dir = args.output_dir

    enhance_features(
        input_dir,
        output_dir,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        unsharp_sigma=args.unsharp_sigma,
        unsharp_amount=args.unsharp_amount,
        unsharp_threshold=args.unsharp_threshold
    )






