import os
import cv2
from tqdm import tqdm


def crop_black_mask_folder(input_root, success_root, error_root, pad=0):
    splits = ["train", "val", "test"]

    for split in splits:
        input_folder = os.path.join(input_root, split)
        success_folder = os.path.join(success_root, split)
        error_folder = os.path.join(error_root, split)

        os.makedirs(success_folder, exist_ok=True)
        os.makedirs(error_folder, exist_ok=True)

        if not os.path.exists(input_folder):
            print(f"Folder not found: {input_folder}")
            continue

        valid_extensions = (".jpg", ".jpeg", ".png")
        image_paths = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(valid_extensions)
        ]

        print(f"\n[{split}] Found {len(image_paths)} images. Starting crop...")

        for img_path in tqdm(image_paths, desc=f"Cropping {split}"):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)

            if img is None:
                continue

            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if not contours:
                    cv2.imwrite(os.path.join(error_folder, img_name), img)
                    continue

                largest_contour = max(contours, key=cv2.contourArea)

                x, y, w, h = cv2.boundingRect(largest_contour)

                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img.shape[1], x + w + pad)
                y2 = min(img.shape[0], y + h + pad)

                cropped = img[y1:y2, x1:x2]

                if cropped.size == 0:
                    cv2.imwrite(os.path.join(error_folder, img_name), img)
                    continue

                cv2.imwrite(os.path.join(success_folder, img_name), cropped)

            except Exception as e:
                print(f"\nError cropping image {img_name}: {e}")
                cv2.imwrite(os.path.join(error_folder, img_name), img)

    print("\n✅ CROP COMPLETED!")
    print(f"- Success: {success_root}")
    print(f"- Error: {error_root}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_ROOT = os.path.join(base_dir, "../outputs/01_black_masks")
    SUCCESS_ROOT = os.path.join(base_dir, "../outputs/02_cropped_receipts")
    ERROR_ROOT = os.path.join(base_dir, "../outputs/02_cropped_receipts_error")

    crop_black_mask_folder(
        input_root=INPUT_ROOT,
        success_root=SUCCESS_ROOT,
        error_root=ERROR_ROOT,
        pad=10
    )
