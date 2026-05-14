from pathlib import Path

DATASET_ROOT = Path("/home/tuan/Desktop/ocr-projects/stage2/raw_datasets_for_rotation")  # sửa lại đường dẫn
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(folder: Path) -> int:
    return sum(
        1 for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )

def get_split_stats(split_dir: Path):
    stats = {}
    if not split_dir.exists():
        return stats

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    for class_dir in class_dirs:
        stats[class_dir.name] = count_images(class_dir)
    return stats

def print_split_stats(split_name: str, stats: dict):
    print(f"\n===== {split_name.upper()} =====")
    total = sum(stats.values())

    if total == 0:
        print("Không có dữ liệu.")
        return

    print(f"{'Class':<10}{'Count':<10}{'Percent'}")
    for cls, count in stats.items():
        pct = count / total * 100
        print(f"{cls:<10}{count:<10}{pct:.2f}%")

    max_count = max(stats.values())
    min_count = min(stats.values())

    if min_count > 0:
        print(f"\nImbalance ratio (max/min): {max_count / min_count:.2f}")
    else:
        print("\nImbalance ratio: INF (có class bằng 0)")

def main():
    train_stats = get_split_stats(DATASET_ROOT / "train")
    val_stats = get_split_stats(DATASET_ROOT / "valid")

    print(f"Dataset root: {DATASET_ROOT}")

    print_split_stats("train", train_stats)
    print_split_stats("valid", val_stats)

    print("\n===== GỢI Ý =====")
    if train_stats:
        max_count = max(train_stats.values())
        min_count = min(train_stats.values())

        if min_count == 0:
            print("- Train đang có class bị thiếu hoàn toàn, cần xử lý ngay.")
        else:
            ratio = max_count / min_count
            if ratio <= 1.5:
                print("- Train khá cân bằng, có thể train bình thường.")
            elif ratio <= 3:
                print("- Train lệch vừa, nên cân nhắc class weights hoặc oversampling.")
            else:
                print("- Train lệch mạnh, nên oversampling/augment thêm class ít hoặc cân bằng lại dữ liệu gốc.")

    print("- Valid chỉ dùng để đo metric trong lúc train.")
    print("- Test có thể tách sau, miễn là là tập unseen hoàn toàn.")

if __name__ == "__main__":
    main()