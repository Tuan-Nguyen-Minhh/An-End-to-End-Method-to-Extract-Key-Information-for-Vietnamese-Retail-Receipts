import os
import cv2
import shutil
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ["train", "val", "test"]


def clear_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")


def build_dataset_pipeline_all_splits(input_root, success_root, error_root, model_path="best.pt", conf=0.5):
    print("Clearing output folders...")
    for split in SPLITS:
        clear_folder(os.path.join(success_root, split))
        clear_folder(os.path.join(error_root, split))

    print(f"Loading model from: {model_path}...")
    model = YOLO(model_path)

    total_success = 0
    total_error = 0

    for split in SPLITS:
        input_folder = os.path.join(input_root, split, "images")
        success_folder = os.path.join(success_root, split)
        error_folder = os.path.join(error_root, split)

        if not os.path.exists(input_folder):
            print(f"[{split}] Input folder not found: {input_folder}")
            continue

        image_paths = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]
        image_paths.sort()

        print(f"\n[{split}] Found {len(image_paths)} images. Starting processing...")

        split_success = 0
        split_error = 0

        for img_path in tqdm(image_paths, desc=f"Processing {split}"):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)

            if img is None:
                split_error += 1
                total_error += 1
                continue

            try:
                result = model.predict(img, conf=conf, verbose=False)[0]

                if result.masks is None or len(result.masks.data) == 0:
                    cv2.imwrite(os.path.join(error_folder, img_name), img)
                    split_error += 1
                    total_error += 1
                    continue

                best_idx = 0
                if result.boxes is not None and len(result.boxes) > 0:
                    best_idx = int(np.argmax(result.boxes.conf.cpu().numpy()))

                mask = result.masks.data[best_idx].cpu().numpy().astype(np.uint8)

                if mask.sum() == 0:
                    cv2.imwrite(os.path.join(error_folder, img_name), img)
                    split_error += 1
                    total_error += 1
                    continue

                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(
                        mask,
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                output = img * mask[:, :, None]

                cv2.imwrite(os.path.join(success_folder, img_name), output)
                split_success += 1
                total_success += 1

            except Exception as e:
                print(f"\n[{split}] Error processing image {img_name}: {e}")
                cv2.imwrite(os.path.join(error_folder, img_name), img)
                split_error += 1
                total_error += 1

        print(f"[{split}] Success: {split_success} | Error: {split_error}")

    print("\nDONE!")
    print(f"- Total success: {total_success}")
    print(f"- Total error  : {total_error}")
    print(f"- Success root : {success_root}")
    print(f"- Error root   : {error_root}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_ROOT = os.path.join(base_dir, "../input_raw_receipts")
    SUCCESS_ROOT = os.path.join(base_dir, "../outputs/01_black_masks")
    ERROR_ROOT = os.path.join(base_dir, "../outputs/01_black_masks_error")
    MODEL_WEIGHTS = os.path.join(base_dir, "../../../models/segmentation_best.pt")

    build_dataset_pipeline_all_splits(
        input_root=INPUT_ROOT,
        success_root=SUCCESS_ROOT,
        error_root=ERROR_ROOT,
        model_path=MODEL_WEIGHTS,
        conf=0.5
    )