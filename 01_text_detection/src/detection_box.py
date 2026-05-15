import os
import glob
import argparse
import logging
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# =========================================================
# DEFAULT PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "../../models/text_detection_yolo_best.pt")

DEFAULT_INPUT_DIRS = {
    "train": os.path.join(BASE_DIR, "../../02_Normalize_Receipts/03_deskewing_and_enhancing/outputs/train/sharpened"),
    "val":   os.path.join(BASE_DIR, "../../02_Normalize_Receipts/03_deskewing_and_enhancing/outputs/val/sharpened"),
    "test":  os.path.join(BASE_DIR, "../../02_Normalize_Receipts/03_deskewing_and_enhancing/outputs/test/sharpened"),
}

DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "../outputs")

# =========================================================
# INFERENCE SETTINGS
# =========================================================
IMG_SIZE           = 1280
CONF_THRES         = 0.08
IOU_THRES          = 0.45
DEFAULT_BATCH_SIZE = 16

# =========================================================
# POST-PROCESS SETTINGS
# =========================================================
EXPAND_X_RATIO     = 0.01   # Avoid clipping into adjacent column
EXPAND_Y_RATIO     = 0.02   # Avoid merging with lines above/below
MIN_BOX_AREA       = 16

MERGE_Y_CENTER_THR = 0.30   # Must be truly Y-aligned to merge
MERGE_X_GAP_THR    = 0.25   # Prevent merging across large whitespace
MERGE_OVERLAP_THR  = 0.50   # If X overlap > 50%, skip manual merge (let NMS handle)

# =========================================================
# FILE EXTENSIONS & LOGGING
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
    """Use glob + set to avoid duplicates (case-insensitive on Windows)."""
    paths = set()
    for ext in IMAGE_EXTS:
        paths.update(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.update(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(paths)


def clip_box(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[int, int, int, int]:
    """
    Clamp coordinates to image boundaries.
    Use w/h (not w-1/h-1) for x2/y2 to avoid losing 1 pixel at the right/bottom edge.
    """
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def expand_box(
    box: list,
    img_w: int,
    img_h: int,
    expand_x_ratio: float = EXPAND_X_RATIO,
    expand_y_ratio: float = EXPAND_Y_RATIO,
) -> list:
    x1, y1, x2, y2, conf, cls_id = box
    bw = x2 - x1
    bh = y2 - y1
    nx1, ny1 = x1 - bw * expand_x_ratio, y1 - bh * expand_y_ratio
    nx2, ny2 = x2 + bw * expand_x_ratio, y2 + bh * expand_y_ratio
    nx1, ny1, nx2, ny2 = clip_box(nx1, ny1, nx2, ny2, img_w, img_h)
    return [nx1, ny1, nx2, ny2, conf, cls_id]


def box_area(box: list) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def should_merge(box_a: list, box_b: list) -> bool:
    """
    Check whether 2 boxes should be merged based on:
      - Same line (y_center close to each other)
      - Close enough horizontally (small x_gap)
      - Not overlapping too much (avoid merging double-detections)
    """
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    max_h = max(ay2 - ay1, by2 - by1)
    if max_h == 0:
        return False

    # Check same line
    y_center_diff = abs((ay1 + ay2) / 2.0 - (by1 + by2) / 2.0)
    if y_center_diff > MERGE_Y_CENTER_THR * max_h:
        return False

    # Check X overlap — if overlap is too large, it's a double-detection, don't merge
    overlap_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    min_w = min(ax2 - ax1, bx2 - bx1)
    if min_w > 0 and overlap_x / min_w > MERGE_OVERLAP_THR:
        return False

    # Check X distance
    if bx1 >= ax2:
        x_gap = bx1 - ax2
    elif ax1 >= bx2:
        x_gap = ax1 - bx2
    else:
        x_gap = 0.0  # Small overlap — still allow merge

    return x_gap <= MERGE_X_GAP_THR * max_h


def merge_two_boxes(box_a: list, box_b: list) -> list:
    """Merge 2 boxes into 1, keep highest confidence."""
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2])
    y2 = max(box_a[3], box_b[3])
    conf = max(box_a[4], box_b[4])
    cls_id = box_a[5] if box_a[4] >= box_b[4] else box_b[5]
    return [x1, y1, x2, y2, conf, cls_id]


def merge_boxes_linewise(boxes: list) -> list:
    """
    Merge boxes line by line.
    Uses a simple union-find algorithm (iterative grow).
    """
    if len(boxes) <= 1:
        return boxes

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    used = [False] * len(boxes)
    merged = []

    for i in range(len(boxes)):
        if used[i]:
            continue
        used[i] = True
        cur = boxes[i]
        changed = True

        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                if should_merge(cur, boxes[j]):
                    cur = merge_two_boxes(cur, boxes[j])
                    used[j] = True
                    changed = True

        merged.append(cur)

    return sorted(merged, key=lambda b: (b[1], b[0]))


def filter_small_boxes(boxes: list, min_area: int = MIN_BOX_AREA) -> list:
    return [b for b in boxes if box_area(b) >= min_area]


def yolo_xyxy_to_txt_line(box: list, img_w: int, img_h: int) -> str:
    xc = min(max((box[0] + box[2]) / 2.0 / img_w, 0.0), 1.0)
    yc = min(max((box[1] + box[3]) / 2.0 / img_h, 0.0), 1.0)
    bw = min(max((box[2] - box[0]) / img_w, 0.0), 1.0)
    bh = min(max((box[3] - box[1]) / img_h, 0.0), 1.0)
    return f"{int(box[5])} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def draw_boxes(
    image: np.ndarray,
    boxes: list,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    vis = image.copy()
    for b in boxes:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thickness)
    return vis


# =========================================================
# INFERENCE & BATCH PROCESSING
# =========================================================

def load_image(image_path: str) -> np.ndarray:
    """Load image as BGR uint8, handle grayscale and BGRA."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def postprocess_boxes(raw_boxes: list, img_w: int, img_h: int) -> list:
    boxes = [expand_box(b, img_w, img_h) for b in raw_boxes]
    boxes = filter_small_boxes(boxes, MIN_BOX_AREA)
    return merge_boxes_linewise(boxes)


def infer_batch(
    model: YOLO, image_paths: list[str], device: str
) -> list[tuple[np.ndarray, list]]:
    """
    Run inference on a batch of images.
    Pre-load images and pass directly to YOLO to avoid dependency on orig_img
    (which is unstable with stream=True on some ultralytics versions).
    """
    imgs = []
    valid_paths = []
    for p in image_paths:
        try:
            imgs.append(load_image(p))
            valid_paths.append(p)
        except ValueError as e:
            logger.warning("Skipping corrupted image: %s", e)

    if not imgs:
        return []

    results = model.predict(
        source=imgs,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=device,
        verbose=False,
    )

    output = []
    for img, result in zip(imgs, results):
        h, w = img.shape[:2]
        raw_boxes = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy    = result.boxes.xyxy.cpu().numpy()
            confs   = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = clip_box(
                    xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], w, h
                )
                raw_boxes.append([x1, y1, x2, y2, float(confs[i]), int(classes[i])])

        output.append((img, postprocess_boxes(raw_boxes, w, h)))

    return output


def save_yolo_label(txt_path: str, boxes: list, img_w: int, img_h: int) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(yolo_xyxy_to_txt_line(b, img_w, img_h) + "\n")


def process_split(
    model: YOLO,
    split_name: str,
    input_dir: str,
    output_root: str,
    device: str,
    batch_size: int,
) -> None:
    image_paths = list_images(input_dir)
    if not image_paths:
        logger.warning("No images found in split '%s': %s", split_name, input_dir)
        return

    out_split_dir = os.path.join(output_root, split_name)
    out_images_dir  = os.path.join(out_split_dir, "images")
    out_labels_dir  = os.path.join(out_split_dir, "labels")
    out_visuals_dir = os.path.join(out_split_dir, "visuals")
    for d in (out_split_dir, out_images_dir, out_labels_dir, out_visuals_dir):
        ensure_dir(d)

    logger.info(
        "Processing split: %s | %d images | input: %s",
        split_name, len(image_paths), input_dir,
    )

    batches = [
        image_paths[i : i + batch_size]
        for i in range(0, len(image_paths), batch_size)
    ]

    with tqdm(total=len(image_paths), desc=split_name, unit="img") as pbar:
        for batch_paths in batches:
            try:
                results = infer_batch(model, batch_paths, device)
            except MemoryError:
                logger.critical(
                    "Out of RAM — stopping. Try reducing --batch (current: %d)", batch_size
                )
                raise
            except Exception as e:
                logger.error("Batch error (skipping %d images): %s", len(batch_paths), e)
                pbar.update(len(batch_paths))
                continue

            for image_path, (img, boxes) in zip(batch_paths, results):
                try:
                    h, w = img.shape[:2]
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    orig_name = os.path.basename(image_path)

                    save_yolo_label(
                        os.path.join(out_labels_dir, base_name + ".txt"), boxes, w, h
                    )
                    shutil.copy2(image_path, os.path.join(out_images_dir, orig_name))
                    cv2.imwrite(
                        os.path.join(out_visuals_dir, base_name + ".jpg"),
                        draw_boxes(img, boxes),
                    )
                except Exception as e:
                    logger.error("FAIL - %s: %s", image_path, e)

                pbar.update(1)


# =========================================================
# ENTRYPOINT
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO text detection inference")
    parser.add_argument("--model",  default=DEFAULT_MODEL_PATH,  help="Path to .pt file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_ROOT, help="Output directory")
    parser.add_argument("--device", default="cpu",               help="Device: 'cpu' or 'cuda:0'")
    parser.add_argument("--batch",  type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Override input dir for split, e.g.: --split train=/data/train",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dirs = dict(DEFAULT_INPUT_DIRS)
    if args.split:
        for item in args.split:
            if "=" in item:
                name, path = item.split("=", 1)
                input_dirs[name] = path

    logger.info("Loading model: %s", args.model)
    model = YOLO(args.model)

    for split_name, input_dir in input_dirs.items():
        process_split(
            model, split_name, input_dir, args.output, args.device, args.batch
        )

    logger.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
