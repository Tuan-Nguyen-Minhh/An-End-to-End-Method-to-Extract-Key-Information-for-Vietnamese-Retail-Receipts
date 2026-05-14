from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
import shutil
import random
import json
import csv
import re

import cv2
import numpy as np


# =========================================================
# CẤU HÌNH
# =========================================================
SOURCE_ROOT = Path("/home/tuan/Desktop/ocr-projects/stage2/raw_rotation_datasets")
OUTPUT_ROOT = Path("/home/tuan/Desktop/ocr-projects/stage2/raw_datasets_for_rotation_balanced_strict_v2")

# Roboflow thường có train / valid / test
SPLIT_NAMES = ["train", "valid", "test"]

CLASSES = ["0", "90", "180", "270"]
CLASS_SET = set(CLASSES)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RANDOM_SEED = 42
OVERWRITE_OUTPUT = False

# Nếu phát hiện khả năng leakage giữa các split thì dừng
ABORT_ON_POTENTIAL_LEAKAGE = False

# Strict augmentation cho train
STRICT_NO_REPEAT_DOCUMENT_PER_TARGET = True

# Leakage check là tùy chọn. Với dataset competition, mặc định tắt để giữ split gốc.
ENABLE_LEAKAGE_CHECK = False

# Nếu bật leakage check thì dùng ngưỡng riêng, mặc định chặt hơn nhiều
# để giảm false positive.
LEAKAGE_PHASH_HAMMING_THRESHOLD = 2

# Ngưỡng gom document group cho train augmentation.
# Dùng 0 để chỉ gom exact-match, tránh over-group các bill chỉ na ná nhau.
TRAIN_GROUP_PHASH_HAMMING_THRESHOLD = 0

# Nếu một ảnh lỗi đọc, dừng luôn cho sạch dữ liệu
FAIL_ON_IMAGE_READ_ERROR = True

# Giới hạn độ dài stem cũ khi đưa vào tên file output
MAX_STEM_LEN = 40


# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class Record:
    record_id: str
    split_name: str
    class_name: str
    angle: int
    path: Path
    canonical_hash_int: int
    canonical_hash_hex: str


# =========================================================
# HÀM TIỆN ÍCH
# =========================================================
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def ensure_clean_output_dir(output_root: Path, overwrite: bool = False) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output folder đã tồn tại và không rỗng: {output_root}\n"
                f"Hãy đổi OUTPUT_ROOT hoặc đặt OVERWRITE_OUTPUT = True."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)


def parse_angle(class_name: str) -> int:
    try:
        angle = int(class_name)
    except ValueError as exc:
        raise ValueError(
            f"Tên class '{class_name}' không hợp lệ. "
            f"Script này chỉ hỗ trợ class: {sorted(CLASSES, key=int)}"
        ) from exc

    if str(angle) not in CLASS_SET:
        raise ValueError(
            f"Class '{class_name}' không nằm trong tập hỗ trợ: {sorted(CLASSES, key=int)}"
        )

    return angle


def safe_name(text: str, max_len: int = MAX_STEM_LEN) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = text.strip("._-")
    if not text:
        text = "img"
    return text[:max_len]


def list_class_dirs(split_dir: Path):
    if not split_dir.exists():
        return []

    valid_dirs = []
    invalid_dirs = []
    for d in split_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name in CLASS_SET:
            valid_dirs.append(d)
        else:
            invalid_dirs.append(d.name)

    if invalid_dirs:
        raise ValueError(
            f"Trong split '{split_dir.name}' có folder không hợp lệ: {sorted(invalid_dirs)}.\n"
            f"Chỉ chấp nhận các class folder: {sorted(CLASSES, key=int)}"
        )

    valid_dirs = sorted(valid_dirs, key=lambda x: int(x.name))
    return valid_dirs


def list_images_in_dir(folder: Path):
    return sorted([p for p in folder.rglob("*") if is_image_file(p)])


def read_image_bgr(image_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    return img


def save_image_bgr(image: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), image)
    if not ok:
        raise IOError(f"Không ghi được ảnh: {save_path}")


def rotate_bgr_by_label_convention(img_bgr: np.ndarray, rotation_angle: int) -> np.ndarray:
    """
    Quy ước:
      0   = ảnh gốc
      90  = xoay 90 độ theo chiều kim đồng hồ
      180 = xoay 180 độ
      270 = xoay 90 độ ngược chiều kim đồng hồ
    """
    rotation_angle = rotation_angle % 360

    if rotation_angle == 0:
        return img_bgr.copy()
    if rotation_angle == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if rotation_angle == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if rotation_angle == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError(f"Góc xoay không hợp lệ: {rotation_angle}")


def compute_required_rotation(source_angle: int, target_angle: int) -> int:
    """
    Với quy ước:
        target = (source + rotation_clockwise) % 360
    => rotation_needed = (target - source) % 360
    """
    return (target_angle - source_angle) % 360


def copy_original_file(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


# =========================================================
# PHASH XOAY-BẤT-BIẾN
# =========================================================
def phash64_from_bgr(img_bgr: np.ndarray) -> int:
    """
    Perceptual hash 64-bit đơn giản:
    - grayscale
    - resize 32x32
    - DCT
    - lấy block 8x8
    - so với median
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    gray = np.float32(gray)

    dct = cv2.dct(gray)
    low = dct[:8, :8].flatten()

    median = np.median(low[1:])
    bits = low > median
    bits[0] = 0

    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)

    return int(hash_value)


def canonical_rotation_invariant_phash(image_path: Path):
    """
    Tính hash bất biến theo 4 góc quay:
    lấy min trong 4 hash để chuẩn hóa.
    """
    img = read_image_bgr(image_path)

    hashes = []
    for angle in (0, 90, 180, 270):
        rotated = rotate_bgr_by_label_convention(img, angle)
        hashes.append(phash64_from_bgr(rotated))

    canonical = min(hashes)
    return canonical, f"{canonical:016x}"


def hamming_distance_64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# =========================================================
# THU THẬP RECORD
# =========================================================
def collect_split_records(split_dir: Path, split_name: str):
    """
    Mỗi record chứa metadata của 1 ảnh.
    """
    if not split_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy split dir: {split_dir}")

    records = []
    seen_classes = set()
    failures = []

    class_dirs = list_class_dirs(split_dir)
    if not class_dirs:
        raise ValueError(f"Không có class folder nào trong: {split_dir}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        angle = parse_angle(class_name)
        seen_classes.add(class_name)

        image_paths = list_images_in_dir(class_dir)
        for idx, img_path in enumerate(image_paths):
            try:
                h_int, h_hex = canonical_rotation_invariant_phash(img_path)
            except Exception as exc:  # noqa: BLE001
                failures.append({
                    "split_name": split_name,
                    "class_name": class_name,
                    "path": str(img_path),
                    "error": repr(exc),
                })
                if FAIL_ON_IMAGE_READ_ERROR:
                    raise
                continue

            records.append(
                Record(
                    record_id=f"{split_name}_{class_name}_{idx:06d}",
                    split_name=split_name,
                    class_name=class_name,
                    angle=angle,
                    path=img_path,
                    canonical_hash_int=h_int,
                    canonical_hash_hex=h_hex,
                )
            )

    missing = CLASS_SET - seen_classes
    if missing:
        print(f"[WARNING] Split {split_name} đang thiếu class: {sorted(missing, key=int)}")

    return records, failures


def group_records_by_class(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec.class_name].append(rec)
    return grouped


def count_records_by_class(records):
    grouped = group_records_by_class(records)
    return {cls: len(grouped.get(cls, [])) for cls in CLASSES}


# =========================================================
# GOM DOCUMENT GROUPS THEO NEAR-DUPLICATE
# =========================================================
def build_document_groups(records, threshold: int):
    """
    Gom các ảnh thành cluster tài liệu dựa trên canonical pHash.
    threshold = 0  -> exact hash
    threshold > 0  -> near-duplicate bằng Hamming distance

    Với vài nghìn ảnh, O(n^2) vẫn chấp nhận được và dễ kiểm soát.
    """
    if not records:
        return {}, {}

    uf = UnionFind(len(records))
    hashes = [rec.canonical_hash_int for rec in records]

    for i in range(len(records)):
        hi = hashes[i]
        for j in range(i + 1, len(records)):
            dist = hamming_distance_64(hi, hashes[j])
            if dist <= threshold:
                uf.union(i, j)

    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        root = uf.find(idx)
        groups[root].append(rec)

    record_to_group = {}
    final_groups = {}
    for group_index, (_, group_records) in enumerate(groups.items()):
        doc_id = f"doc_{group_index:06d}"
        final_groups[doc_id] = group_records
        for rec in group_records:
            record_to_group[rec.record_id] = doc_id

    return final_groups, record_to_group


# =========================================================
# SOÁT LEAKAGE GIỮA CÁC SPLIT
# =========================================================
def detect_pairwise_leakage(records_a, records_b, threshold: int, split_a: str, split_b: str):
    """
    Soát overlap bằng canonical pHash + Hamming distance.
    threshold nhỏ sẽ bắt được các case giống nhau nhưng crop / nén hơi khác.
    """
    rows = []
    matched_a = set()
    matched_b = set()
    match_keys = set()

    for rec_a in records_a:
        for rec_b in records_b:
            dist = hamming_distance_64(rec_a.canonical_hash_int, rec_b.canonical_hash_int)
            if dist <= threshold:
                key = tuple(sorted((rec_a.record_id, rec_b.record_id)))
                if key in match_keys:
                    continue
                match_keys.add(key)
                matched_a.add(rec_a.record_id)
                matched_b.add(rec_b.record_id)
                rows.append({
                    "split_a": split_a,
                    "record_id_a": rec_a.record_id,
                    "class_a": rec_a.class_name,
                    "path_a": str(rec_a.path),
                    "split_b": split_b,
                    "record_id_b": rec_b.record_id,
                    "class_b": rec_b.class_name,
                    "path_b": str(rec_b.path),
                    "hamming_distance": dist,
                    "hash_a": rec_a.canonical_hash_hex,
                    "hash_b": rec_b.canonical_hash_hex,
                })

    summary = {
        "split_a": split_a,
        "split_b": split_b,
        "matched_pairs": len(rows),
        "items_involved_a": len(matched_a),
        "items_involved_b": len(matched_b),
        "threshold": threshold,
        "rows": rows,
    }
    return summary


def detect_all_split_leakage(records_by_split, threshold: int):
    split_names = [name for name in SPLIT_NAMES if name in records_by_split]
    pair_summaries = []
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a = split_names[i]
            b = split_names[j]
            pair_summaries.append(
                detect_pairwise_leakage(records_by_split[a], records_by_split[b], threshold, a, b)
            )

    total_pairs = sum(s["matched_pairs"] for s in pair_summaries)
    return {
        "threshold": threshold,
        "pair_summaries": pair_summaries,
        "total_matched_pairs": total_pairs,
    }


def save_leakage_report(output_root: Path, leakage_summary: dict):
    json_path = output_root / "potential_leakage_report.json"
    csv_path = output_root / "potential_leakage_pairs.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(leakage_summary, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split_a",
                "record_id_a",
                "class_a",
                "path_a",
                "split_b",
                "record_id_b",
                "class_b",
                "path_b",
                "hamming_distance",
                "hash_a",
                "hash_b",
            ],
        )
        writer.writeheader()
        for pair_summary in leakage_summary["pair_summaries"]:
            for row in pair_summary["rows"]:
                writer.writerow(row)


# =========================================================
# COPY SPLIT GỐC + MANIFEST
# =========================================================
def copy_split_original(records, output_split_dir: Path, record_to_group_id: dict, manifest_rows: list):
    """
    Copy nguyên ảnh gốc sang dataset mới.
    Vừa copy vừa ghi manifest để sau này trace ngược được.
    """
    grouped = group_records_by_class(records)
    copied_counts = {}

    for class_name in CLASSES:
        class_records = grouped.get(class_name, [])
        dst_class_dir = output_split_dir / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for idx, rec in enumerate(class_records):
            src_path = rec.path
            ext = src_path.suffix.lower()
            stem = safe_name(src_path.stem)
            dst_name = f"orig_{stem}_{idx:06d}{ext}"
            dst_path = dst_class_dir / dst_name
            copy_original_file(src_path, dst_path)
            copied += 1

            manifest_rows.append({
                "row_type": "original_copy",
                "split_name": rec.split_name,
                "target_class": rec.class_name,
                "source_class": rec.class_name,
                "rotation_applied": 0,
                "source_path": str(src_path),
                "output_path": str(dst_path),
                "record_id": rec.record_id,
                "document_group_id": record_to_group_id.get(rec.record_id, ""),
                "canonical_hash_hex": rec.canonical_hash_hex,
            })

        copied_counts[class_name] = copied

    return copied_counts


# =========================================================
# BUILD SOURCE GROUPS THEO DOCUMENT UNIQUE (TRAIN ONLY)
# =========================================================
def build_train_document_groups(train_records, threshold: int):
    groups, record_to_group = build_document_groups(train_records, threshold)
    return groups, record_to_group


def choose_best_source_record_for_target(group_records, target_class, source_class_usage, rng):
    """
    Chọn source record cho 1 document-group để sinh target_class.

    Tiêu chí:
    1) ưu tiên source_class != target_class (tránh xoay 0 độ)
    2) ưu tiên source class đang ít được dùng hơn cho target này
    3) nếu hòa thì random có seed
    """
    candidates = [r for r in group_records if r.class_name != target_class]
    if not candidates:
        return None

    min_usage = min(source_class_usage[r.class_name] for r in candidates)
    best = [r for r in candidates if source_class_usage[r.class_name] == min_usage]

    return rng.choice(best)


def augment_train_strict(train_records, output_train_dir: Path, rng: random.Random, threshold: int, manifest_rows: list):
    """
    Cân bằng train tới majority class theo kiểu strict:
    - chỉ dùng train làm nguồn
    - mỗi document unique chỉ dùng tối đa 1 lần cho mỗi target class
    - tránh source cùng class target
    - nếu không đủ document unique thì dừng
    """
    original_counts = count_records_by_class(train_records)
    majority_count = max(original_counts.values())

    doc_groups, train_record_to_group = build_train_document_groups(train_records, threshold)
    unique_doc_count = len(doc_groups)

    generated_counts = {cls: 0 for cls in CLASSES}

    for target_class in CLASSES:
        target_angle = parse_angle(target_class)
        current_count = original_counts[target_class]
        need = majority_count - current_count

        if need <= 0:
            continue

        eligible_doc_ids = []
        for doc_id, group in doc_groups.items():
            if any(r.class_name != target_class for r in group):
                eligible_doc_ids.append(doc_id)

        if STRICT_NO_REPEAT_DOCUMENT_PER_TARGET and len(eligible_doc_ids) < need:
            raise RuntimeError(
                f"Không đủ document unique để cân bằng target class {target_class} theo chế độ strict.\n"
                f"Need = {need}, eligible_unique_documents = {len(eligible_doc_ids)}.\n"
                f"Bạn có thể:\n"
                f"1) giữ strict và chấp nhận không balance tuyệt đối,\n"
                f"2) hoặc tắt STRICT_NO_REPEAT_DOCUMENT_PER_TARGET để cho phép reuse."
            )

        rng.shuffle(eligible_doc_ids)

        used_doc_ids_for_target = set()
        source_class_usage = defaultdict(int)
        aug_index = 0

        for doc_id in eligible_doc_ids:
            if generated_counts[target_class] >= need:
                break
            if doc_id in used_doc_ids_for_target:
                continue

            source_record = choose_best_source_record_for_target(
                doc_groups[doc_id],
                target_class,
                source_class_usage,
                rng,
            )

            if source_record is None:
                continue

            src_path = source_record.path
            src_class = source_record.class_name
            src_angle = source_record.angle
            rotation_needed = compute_required_rotation(src_angle, target_angle)

            if rotation_needed == 0:
                continue

            img = read_image_bgr(src_path)
            aug_img = rotate_bgr_by_label_convention(img, rotation_needed)

            ext = src_path.suffix.lower()
            stem = safe_name(src_path.stem)
            dst_class_dir = output_train_dir / target_class
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            dst_name = (
                f"aug_target_{target_class}_"
                f"from_{src_class}_"
                f"rot_{rotation_needed}_"
                f"doc_{doc_id[-6:]}_"
                f"{stem}_"
                f"{aug_index:06d}{ext}"
            )
            dst_path = dst_class_dir / dst_name
            save_image_bgr(aug_img, dst_path)

            manifest_rows.append({
                "row_type": "augmentation",
                "split_name": "train",
                "target_class": target_class,
                "source_class": src_class,
                "rotation_applied": rotation_needed,
                "source_path": str(src_path),
                "output_path": str(dst_path),
                "record_id": source_record.record_id,
                "document_group_id": train_record_to_group.get(source_record.record_id, doc_id),
                "canonical_hash_hex": source_record.canonical_hash_hex,
            })

            generated_counts[target_class] += 1
            used_doc_ids_for_target.add(doc_id)
            source_class_usage[src_class] += 1
            aug_index += 1

        if STRICT_NO_REPEAT_DOCUMENT_PER_TARGET and generated_counts[target_class] < need:
            raise RuntimeError(
                f"Target class {target_class} chưa đủ sau augment strict.\n"
                f"Need = {need}, generated = {generated_counts[target_class]}.\n"
                f"Kiểm tra lại dữ liệu gốc hoặc cân nhắc nới điều kiện."
            )

    final_counts = {cls: original_counts[cls] + generated_counts[cls] for cls in CLASSES}

    balance_info = {
        "original_counts": original_counts,
        "generated_counts": generated_counts,
        "final_counts": final_counts,
        "majority_count": majority_count,
        "unique_train_documents_by_group": unique_doc_count,
    }

    return balance_info


def save_manifest(output_root: Path, manifest_rows):
    csv_path = output_root / "dataset_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_type",
                "split_name",
                "target_class",
                "source_class",
                "rotation_applied",
                "source_path",
                "output_path",
                "record_id",
                "document_group_id",
                "canonical_hash_hex",
            ],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)


def save_failures(output_root: Path, failures: list):
    if not failures:
        return
    json_path = output_root / "read_failures.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)


def save_final_report(output_root: Path, report: dict):
    report_path = output_root / "augmentation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# =========================================================
# MAIN
# =========================================================
def main():
    rng = random.Random(RANDOM_SEED)

    ensure_clean_output_dir(OUTPUT_ROOT, overwrite=OVERWRITE_OUTPUT)

    print(f"Source root: {SOURCE_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")

    records_by_split = {}
    failures_all = []

    print("\n[1/7] Thu thập records + tính canonical hash cho từng split...")
    for split_name in SPLIT_NAMES:
        split_src = SOURCE_ROOT / split_name
        if not split_src.exists():
            print(f"[INFO] Bỏ qua split không tồn tại: {split_src}")
            continue

        records, failures = collect_split_records(split_src, split_name)
        records_by_split[split_name] = records
        failures_all.extend(failures)
        print(f"  - {split_name}: {len(records)} ảnh | counts = {count_records_by_class(records)}")

    if "train" not in records_by_split:
        raise FileNotFoundError("Không tìm thấy split train. Không thể tiếp tục.")
    if "valid" not in records_by_split:
        raise FileNotFoundError("Không tìm thấy split valid. Không thể tiếp tục.")

    save_failures(OUTPUT_ROOT, failures_all)

    print("[2/7] Leakage check giữa các split...")
    leakage_summary = None
    if ENABLE_LEAKAGE_CHECK:
        leakage_summary = detect_all_split_leakage(records_by_split, LEAKAGE_PHASH_HAMMING_THRESHOLD)
        save_leakage_report(OUTPUT_ROOT, leakage_summary)

        for pair_summary in leakage_summary["pair_summaries"]:
            print(
                f"  - {pair_summary['split_a']} vs {pair_summary['split_b']}: "
                f"matched_pairs = {pair_summary['matched_pairs']}"
            )

        if leakage_summary["total_matched_pairs"] > 0:
            print(
                "[WARNING] Có một số cặp ảnh tương tự giữa các split. "
                "Đây là cảnh báo heuristic, không phải kết luận data leakage."
            )
            print(f"Xem file: {OUTPUT_ROOT / 'potential_leakage_pairs.csv'}")
            print(f"Xem file: {OUTPUT_ROOT / 'potential_leakage_report.json'}")

            if ABORT_ON_POTENTIAL_LEAKAGE:
                raise RuntimeError(
                    "Dừng script vì phát hiện các cặp tương tự giữa các split. "
                    "Hãy kiểm tra split gốc trước khi augment."
                )
    else:
        print("  - Bỏ qua leakage check, giữ nguyên split gốc của dataset competition.")

    print("\n[3/7] Gán document group cho từng split để trace về sau...")
    group_summary_by_split = {}
    record_to_group_by_split = {}
    for split_name, records in records_by_split.items():
        groups, record_to_group = build_document_groups(records, TRAIN_GROUP_PHASH_HAMMING_THRESHOLD)
        group_summary_by_split[split_name] = {
            "num_records": len(records),
            "num_document_groups": len(groups),
        }
        record_to_group_by_split[split_name] = record_to_group
        print(
            f"  - {split_name}: {len(records)} records -> {len(groups)} document groups"
        )

    manifest_rows = []

    print("\n[4/7] Copy valid / test nguyên trạng sang dataset mới...")
    copied_counts_by_split = {}
    for split_name in ["valid", "test"]:
        if split_name not in records_by_split:
            continue
        output_split = OUTPUT_ROOT / split_name
        output_split.mkdir(parents=True, exist_ok=True)
        copied_counts_by_split[split_name] = copy_split_original(
            records_by_split[split_name],
            output_split,
            record_to_group_by_split[split_name],
            manifest_rows,
        )
        print(f"  - copied {split_name}: {copied_counts_by_split[split_name]}")

    print("\n[5/7] Copy train gốc sang dataset mới...")
    output_train = OUTPUT_ROOT / "train"
    output_train.mkdir(parents=True, exist_ok=True)
    train_copied_counts = copy_split_original(
        records_by_split["train"],
        output_train,
        record_to_group_by_split["train"],
        manifest_rows,
    )
    print(f"  - copied train: {train_copied_counts}")

    print("\n[6/7] Augment train theo chế độ strict để cân bằng class...")
    balance_info = augment_train_strict(
        records_by_split["train"],
        output_train,
        rng,
        TRAIN_GROUP_PHASH_HAMMING_THRESHOLD,
        manifest_rows,
    )
    save_manifest(OUTPUT_ROOT, manifest_rows)

    print("\n[7/7] Ghi report cuối cùng...")
    if leakage_summary is None:
        potential_leakage_summary = {
            "enabled": False,
            "message": "Leakage check bị tắt để giữ nguyên split gốc của dataset competition.",
        }
    else:
        potential_leakage_summary = {
            "enabled": True,
            "threshold": leakage_summary["threshold"],
            "total_matched_pairs": leakage_summary["total_matched_pairs"],
            "pair_summaries": [
                {
                    "split_a": p["split_a"],
                    "split_b": p["split_b"],
                    "matched_pairs": p["matched_pairs"],
                    "items_involved_a": p["items_involved_a"],
                    "items_involved_b": p["items_involved_b"],
                }
                for p in leakage_summary["pair_summaries"]
            ],
        }

    final_report = {
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "rotation_convention": {
            "0": "original",
            "90": "clockwise 90 degrees",
            "180": "rotate 180 degrees",
            "270": "counterclockwise 90 degrees",
        },
        "settings": {
            "random_seed": RANDOM_SEED,
            "enable_leakage_check": ENABLE_LEAKAGE_CHECK,
            "abort_on_potential_leakage": ABORT_ON_POTENTIAL_LEAKAGE,
            "strict_no_repeat_document_per_target": STRICT_NO_REPEAT_DOCUMENT_PER_TARGET,
            "leakage_phash_hamming_threshold": LEAKAGE_PHASH_HAMMING_THRESHOLD,
            "train_group_phash_hamming_threshold": TRAIN_GROUP_PHASH_HAMMING_THRESHOLD,
            "fail_on_image_read_error": FAIL_ON_IMAGE_READ_ERROR,
        },
        "split_counts_before": {
            split_name: count_records_by_class(records)
            for split_name, records in records_by_split.items()
        },
        "copied_counts": {
            "train": train_copied_counts,
            **copied_counts_by_split,
        },
        "potential_leakage_summary": potential_leakage_summary,
        "document_group_summary_by_split": group_summary_by_split,
        "balance_info": balance_info,
        "num_manifest_rows": len(manifest_rows),
        "notes": [
            "valid và test được copy nguyên trạng, không augment",
            "augment chỉ dùng train",
            "mỗi document unique chỉ dùng tối đa 1 lần cho mỗi target class ở chế độ strict",
            "tránh dùng source cùng class target để không tạo bản xoay 0 độ vô nghĩa",
            "leakage check mặc định bị tắt; dataset được giữ nguyên theo split gốc từ nguồn competition",
            "train document grouping dùng exact-match để tránh over-group các bill chỉ na ná nhau",
            "nếu cần leakage check lại, hãy bật ENABLE_LEAKAGE_CHECK và xem đó là cảnh báo heuristic, không phải chứng minh tuyệt đối",
            "dataset_manifest.csv lưu cả ảnh gốc lẫn ảnh augment để trace ngược nguồn dữ liệu",
        ],
    }
    save_final_report(OUTPUT_ROOT, final_report)

    print("\n===== HOÀN TẤT =====")
    print("Train gốc:")
    for cls in CLASSES:
        print(f"  class {cls}: {balance_info['original_counts'][cls]}")

    print("\nTrain sinh thêm:")
    for cls in CLASSES:
        print(f"  class {cls}: {balance_info['generated_counts'][cls]}")

    print("\nTrain sau cân bằng:")
    for cls in CLASSES:
        print(f"  class {cls}: {balance_info['final_counts'][cls]}")

    print(f"\nMajority target: {balance_info['majority_count']}")
    print(f"Unique train documents by group: {balance_info['unique_train_documents_by_group']}")
    print(f"Manifest: {OUTPUT_ROOT / 'dataset_manifest.csv'}")
    print(f"Leakage:  {OUTPUT_ROOT / 'potential_leakage_pairs.csv'}")
    print(f"Report:   {OUTPUT_ROOT / 'augmentation_report.json'}")


if __name__ == "__main__":
    main()
