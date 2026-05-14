import os
import csv
import shutil
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# =========================
# CONFIG
# =========================
base_dir = Path(__file__).resolve().parent.parent
INPUT_DIR  = base_dir / "input_rotated_receipts"
OUTPUT_DIR = base_dir / "outputs"

# Automatically select number of workers based on CPU count
NUM_WORKERS = min(8, max(1, multiprocessing.cpu_count() - 1))

# If True, only save the final sharpened image (skip deskewed/gray/clahe intermediate results)
SAVE_ONLY_FINAL = True

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =========================
# OUTPUT STRUCTURE
# =========================
# OUTPUT_DIR/
# ├── deskewed/          <- deskewed image (BGR)       [only if SAVE_ONLY_FINAL=False]
# ├── gray/              <- grayscale                  [only if SAVE_ONLY_FINAL=False]
# ├── clahe/             <- after CLAHE                [only if SAVE_ONLY_FINAL=False]
# ├── sharpened/         <- final result               [always saved]
# ├── errors/            <- original error images (copied here for review)
# └── processing_log.csv <- full processing log


# =========================
# CORE FUNCTIONS
# =========================
def rotate_keep_canvas(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def estimate_angle_from_mask(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    angle = cv2.minAreaRect(cnt)[-1]

    if angle < -45:
        angle = 90 + angle

    return float(angle)


def projection_score(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_sum = np.sum(binary > 0, axis=1).astype(np.float32)
    return float(np.var(row_sum))


def refine_angle_by_projection(
    bgr: np.ndarray,
    coarse_angle: float,
    search_range: float = 2.0,
    step: float = 0.2,
) -> tuple[float, float]:
    n_steps = int(round(2 * search_range / step)) + 1
    angles  = np.linspace(coarse_angle - search_range, coarse_angle + search_range, n_steps)

    best_angle = coarse_angle
    best_score = -1.0

    for angle in angles:
        rotated = rotate_keep_canvas(bgr, float(angle))
        gray    = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        score   = projection_score(gray)
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return best_angle, best_score


def preprocess_receipt(image_path: Path) -> dict:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    coarse_angle              = estimate_angle_from_mask(bgr)
    refined_angle, best_score = refine_angle_by_projection(bgr, coarse_angle)

    deskewed_bgr = rotate_keep_canvas(bgr, refined_angle)
    gray         = cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2GRAY)

    clahe_op  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe_op.apply(gray)

    blur      = cv2.GaussianBlur(clahe_img, (0, 0), 1.0)
    sharpened = cv2.addWeighted(clahe_img, 1.5, blur, -0.5, 0)

    return {
        "deskewed_bgr":  deskewed_bgr,
        "gray":          gray,
        "clahe":         clahe_img,
        "sharpened":     sharpened,
        "coarse_angle":  coarse_angle,
        "refined_angle": refined_angle,
        "best_score":    best_score,
    }


# =========================
# SAVE HELPERS
# =========================
def ensure_dirs(out: Path) -> None:
    subdirs = ["sharpened", "errors"]
    if not SAVE_ONLY_FINAL:
        subdirs += ["deskewed", "gray", "clahe"]
    for d in subdirs:
        (out / d).mkdir(parents=True, exist_ok=True)


def _out(output_dir: Path, name: str, rel_parent: Path) -> Path:
    d = output_dir / name / rel_parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_results(image_path: Path, results: dict, input_dir: Path, output_dir: Path) -> None:
    rel    = image_path.relative_to(input_dir)
    stem   = rel.stem
    ext    = rel.suffix
    parent = rel.parent

    if not SAVE_ONLY_FINAL:
        cv2.imwrite(str(_out(output_dir, "deskewed", parent) / f"{stem}{ext}"), results["deskewed_bgr"])
        cv2.imwrite(str(_out(output_dir, "gray",     parent) / f"{stem}{ext}"), results["gray"])
        cv2.imwrite(str(_out(output_dir, "clahe",    parent) / f"{stem}{ext}"), results["clahe"])

    cv2.imwrite(str(_out(output_dir, "sharpened", parent) / f"{stem}{ext}"), results["sharpened"])


def save_error_image(image_path: Path, input_dir: Path, output_dir: Path) -> None:
    """Copy original error image into errors/ folder for later review."""
    rel      = image_path.relative_to(input_dir)
    dest_dir = output_dir / "errors" / rel.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(image_path), str(dest_dir / image_path.name))


# =========================
# WORKER
# =========================
def process_one_image(args: tuple) -> dict:
    image_path_str, input_dir_str, output_dir_str = args
    image_path = Path(image_path_str)
    input_dir  = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    try:
        results = preprocess_receipt(image_path)
        save_results(image_path, results, input_dir, output_dir)
        return {
            "image_path":    image_path_str,
            "status":        "ok",
            "coarse_angle":  round(results["coarse_angle"],  2),
            "refined_angle": round(results["refined_angle"], 2),
            "best_score":    round(results["best_score"],    4),
            "error":         "",
        }
    except Exception as e:
        try:
            save_error_image(image_path, input_dir, output_dir)
        except Exception:
            pass
        return {
            "image_path":    image_path_str,
            "status":        "error",
            "coarse_angle":  "",
            "refined_angle": "",
            "best_score":    "",
            "error":         str(e),
        }


# =========================
# UTILITIES
# =========================
def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def save_csv_log(records: list[dict], csv_path: Path) -> None:
    fieldnames = ["image_path", "status", "coarse_angle", "refined_angle", "best_score", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_dirs(OUTPUT_DIR)

    image_paths = collect_images(INPUT_DIR)
    total       = len(image_paths)
    print(f"Found {total} images  |  Workers: {NUM_WORKERS}")

    if total == 0:
        print("No images found. Check INPUT_DIR.")
        return

    tasks: list[tuple] = [(str(p), str(INPUT_DIR), str(OUTPUT_DIR)) for p in image_paths]
    all_records: list[dict] = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_one_image, t): t for t in tasks}

        iterator = (
            tqdm(as_completed(futures), total=total, desc="Processing")
            if HAS_TQDM else as_completed(futures)
        )

        for idx, future in enumerate(iterator, start=1):
            result = future.result()
            all_records.append(result)

            if not HAS_TQDM:
                name = Path(result["image_path"]).name
                if result["status"] == "ok":
                    print(f"[{idx}/{total}] OK    | {name} | coarse={result['coarse_angle']} | refined={result['refined_angle']}")
                else:
                    print(f"[{idx}/{total}] ERROR | {name} | {result['error']}")

    csv_path = OUTPUT_DIR / "processing_log.csv"
    save_csv_log(all_records, csv_path)

    ok_count  = sum(r["status"] == "ok"    for r in all_records)
    err_count = sum(r["status"] == "error" for r in all_records)

    print("\n========== DONE ==========")
    print(f"Success : {ok_count}")
    print(f"Error   : {err_count}")
    if err_count:
        print(f"  -> Error images copied to: {OUTPUT_DIR / 'errors'}/")
    print(f"Log CSV : {csv_path}")
    print(f"Output  : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()