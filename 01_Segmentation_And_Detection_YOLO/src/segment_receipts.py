import os
import glob
import argparse
import logging

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# =========================================================
# DEFAULT PATHS  (relative to this script's location)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_PATH  = os.path.join(BASE_DIR, "../../models/bill_segmentation_best.pt")
DEFAULT_INPUT_ROOT  = os.path.join(BASE_DIR, "../inputs")   # train / val / test sub-dirs
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "../outputs")

# =========================================================
# INFERENCE SETTINGS
# =========================================================
CONF_THRES = 0.5

# =========================================================
# FILE EXTENSIONS
# =========================================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =========================================================
# UTILS
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(folder: str) -> list[str]:
    paths = set()
    for ext in IMAGE_EXTS:
        paths.update(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.update(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(paths)


# =========================================================
# CORE: process one image → binary mask + detection visual
# =========================================================

def process_image(model: YOLO, image_path: str,
                  out_masks_dir: str, out_visuals_dir: str) -> bool:
    """
    Runs YOLO segmentation on a single image and saves:
      1. binary_mask  — white receipt on black background (grayscale PNG)
      2. detection_visual — original image with segmentation contour +
                            bounding box drawn on top (BGR JPEG)

    Returns True on success, False if no mask was found.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Cannot read image: %s", image_path)
        return False

    result = model.predict(img, conf=CONF_THRES, verbose=False)[0]

    if result.masks is None or len(result.masks.data) == 0:
        logger.warning("No mask found for: %s", os.path.basename(image_path))
        return False

    # Pick the instance with the highest confidence
    best_idx = 0
    if result.boxes is not None and len(result.boxes) > 0:
        best_idx = int(np.argmax(result.boxes.conf.cpu().numpy()))

    # ── Binary instance mask ─────────────────────────────
    instance_mask = result.masks.data[best_idx].cpu().numpy().astype(np.uint8)

    if instance_mask.sum() == 0:
        logger.warning("Empty mask for: %s", os.path.basename(image_path))
        return False

    # Resize mask to original image dimensions if needed
    if instance_mask.shape[:2] != img.shape[:2]:
        instance_mask = cv2.resize(
            instance_mask,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # ── 1. Save binary mask (white on black) ─────────────
    binary_mask_img = (instance_mask * 255).astype(np.uint8)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(out_masks_dir, base_name + ".jpg"), binary_mask_img)

    # ── 2. Build detection visual ─────────────────────────
    # Draw segmentation contour + bounding box on the ORIGINAL image
    contour_binary = binary_mask_img.copy()
    contours, _ = cv2.findContours(
        contour_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detection_vis = img.copy()

    if contours:
        largest = max(contours, key=cv2.contourArea)

        # Segmentation contour in green
        cv2.drawContours(detection_vis, [largest], -1, (0, 255, 0), 2)

        # Bounding box in blue
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(detection_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Confidence label
        conf_val = float(result.boxes.conf.cpu().numpy()[best_idx])
        label = f"receipt  {conf_val:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(detection_vis, (x, y - th - 6), (x + tw + 4, y), (255, 0, 0), -1)
        cv2.putText(detection_vis, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(out_visuals_dir, base_name + ".jpg"), detection_vis)
    return True


# =========================================================
# BATCH PROCESSING
# =========================================================

def process_split(model: YOLO, split_name: str,
                  input_dir: str, output_root: str) -> None:
    image_paths = list_images(input_dir)
    if not image_paths:
        logger.warning("No images found in split '%s': %s", split_name, input_dir)
        return

    out_split      = os.path.join(output_root, split_name)
    out_masks_dir  = os.path.join(out_split, "binary_masks")
    out_visuals_dir = os.path.join(out_split, "detection_visuals")
    for d in (out_split, out_masks_dir, out_visuals_dir):
        ensure_dir(d)

    logger.info("Processing split: %s | %d images", split_name, len(image_paths))

    ok = skipped = 0
    for image_path in tqdm(image_paths, desc=split_name, unit="img"):
        try:
            success = process_image(model, image_path, out_masks_dir, out_visuals_dir)
            if success:
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            logger.error("FAIL - %s: %s", image_path, e)
            skipped += 1

    logger.info("Split '%s' done — OK: %d | Skipped: %d", split_name, ok, skipped)


# =========================================================
# ENTRYPOINT
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO segmentation: produce binary masks + detection visuals"
    )
    parser.add_argument("--model",  default=DEFAULT_MODEL_PATH,  help="Path to .pt model")
    parser.add_argument("--input",  default=DEFAULT_INPUT_ROOT,  help="Root input directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_ROOT, help="Root output directory")
    parser.add_argument("--conf",   type=float, default=CONF_THRES, help="Confidence threshold")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Sub-directory split names to process (default: train val test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Loading model: %s", args.model)
    model = YOLO(args.model)

    for split in args.splits:
        split_input = os.path.join(args.input, split)
        if os.path.isdir(split_input):
            process_split(model, split, split_input, args.output)
        else:
            logger.warning("Split directory not found, skipping: %s", split_input)

    logger.info("All done. Outputs saved to: %s", args.output)


if __name__ == "__main__":
    main()
