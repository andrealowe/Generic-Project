#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil

# allowed image extensions
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".bmp")

def estimate_background(img, kernel_size=51):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def denoise_and_flatten(input_dir: str,
                        output_dir: str,
                        median_ksize: int,
                        bilateral_d: int,
                        bilateral_sigmaColor: int,
                        bilateral_sigmaSpace: int,
                        bg_kernel: int):
    """
    Denoise & subtract background from sonar images, preserving class subdirs:
      - median filter
      - bilateral filter
      - background estimation & subtraction
    Writes outputs into mirrored class subdirs under output_dir.
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
    total = 0
    for cls in class_dirs:
        cls_path = os.path.join(input_dir, cls)
        imgs = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        total += len(imgs)
    print(f"[INFO] Found {total} image(s) across {len(class_dirs)} class folder(s) in '{input_dir}'")

    processed = 0
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

            # median filter
            den = cv2.medianBlur(img, median_ksize)
            # bilateral filter
            den = cv2.bilateralFilter(
                den,
                bilateral_d,
                bilateral_sigmaColor,
                bilateral_sigmaSpace
            )
            # background subtraction
            bg = estimate_background(den, kernel_size=bg_kernel)
            flat = cv2.subtract(den, bg)
            flat = np.clip(flat, 0, 255).astype(np.uint8)

            dst_path = os.path.join(dst_cls, fname)
            cv2.imwrite(dst_path, flat)
            processed += 1

    print(f"[INFO] Denoised & background-subtracted {processed} image(s) into '{output_dir}'")
    
    if is_flow:
        # Write outputs into the named output folder
        zip_base = "/workflow/outputs/denoised_images"
        shutil.make_archive(zip_base, 'zip', output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Denoise & subtract background (preserving class subdirs), local or in a Flow."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=f"/domino/datasets/local/{os.environ['DOMINO_PROJECT_NAME']}/cleaned_data/normalized",
        help="Local input (step1 output); Flow jobs read `/workflow/inputs/input_dir`"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"/domino/datasets/local/{os.environ['DOMINO_PROJECT_NAME']}/cleaned_data/noise_removal",
        help="Local output; Flow jobs write to `/workflow/outputs/denoised_sonar_images`"
    )
    parser.add_argument("--median_ksize", type=int, default=5,
                        help="kernel size for median blur")
    parser.add_argument("--bilateral_d", type=int, default=9,
                        help="diameter for bilateral filter")
    parser.add_argument("--bilateral_sigmaColor", type=int, default=75,
                        help="sigmaColor for bilateral filter")
    parser.add_argument("--bilateral_sigmaSpace", type=int, default=75,
                        help="sigmaSpace for bilateral filter")
    parser.add_argument("--bg_kernel", type=int, default=51,
                        help="kernel size for background estimation")
    args = parser.parse_args()

    # detect Flow mode
    is_flow = os.getenv("DOMINO_IS_WORKFLOW_JOB", "false").lower() == "true"

    if is_flow:
        # Flow‐mounted input & output paths
        input_zip = "/workflow/inputs/normalized_images"
        tmp_in = "/tmp/normalized"
        shutil.unpack_archive(input_zip, tmp_in, 'zip')
        input_dir = tmp_in
        output_dir = "/tmp/denoised"
    
    else:
        # Local development paths
        input_dir  = args.input_dir
        output_dir = args.output_dir

    denoise_and_flatten(
        input_dir,
        output_dir,
        median_ksize=args.median_ksize,
        bilateral_d=args.bilateral_d,
        bilateral_sigmaColor=args.bilateral_sigmaColor,
        bilateral_sigmaSpace=args.bilateral_sigmaSpace,
        bg_kernel=args.bg_kernel
    )