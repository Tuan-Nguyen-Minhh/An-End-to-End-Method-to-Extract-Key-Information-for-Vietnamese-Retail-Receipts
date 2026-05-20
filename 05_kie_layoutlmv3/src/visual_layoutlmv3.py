import os
import json
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
from matplotlib import font_manager
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification


# ==========================================
# 1. CONFIG
# ==========================================
TARGET_IMAGE = ""
# Nếu muốn test ảnh cụ thể:
# TARGET_IMAGE = "your_image_name.jpg"

IMG_FOLDER = "/home/tuan/Desktop/ocr-projects/stage8_done_ocr/ocr_done/train/images"
JSON_FOLDER = "/home/tuan/Desktop/ocr-projects/stage8_done_ocr/ocr_done/train/jsons"

MODEL_DIR = "/home/tuan/Desktop/ocr-projects/stage9_KIE/LayoutLMv3/best_model"

SAVE_FOLDER = "./results_layoutlmv3_report"
os.makedirs(SAVE_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Report options
REPORT_MODE = True
SHOW_CONF_ON_BOX = False
CONF_THRESHOLD = 0.50
SAVE_PDF = True
SAVE_DPI = 300

# Nếu True: chỉ vẽ main field, PREFIX đưa vào bảng
HIDE_PREFIX_ON_IMAGE = True


# ==========================================
# 2. FIX VIETNAMESE FONT
# ==========================================
def setup_vietnamese_font():
    """
    Fix lỗi tiếng Việt khi vẽ bằng matplotlib và khi export PDF.
    Ưu tiên các font thường có sẵn trên Ubuntu/Linux.
    """

    font_candidates = [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]

    selected_font_path = None

    for fp in font_candidates:
        if os.path.exists(fp):
            selected_font_path = fp
            break

    if selected_font_path is None:
        print("Không tìm thấy font tiếng Việt tốt. Dùng font mặc định của matplotlib.")
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        return None

    font_manager.fontManager.addfont(selected_font_path)
    font_prop = font_manager.FontProperties(fname=selected_font_path)
    font_name = font_prop.get_name()

    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    # Quan trọng khi save PDF: embed font TrueType, tránh lỗi dấu tiếng Việt
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    print("Using Vietnamese font:", selected_font_path)
    print("Font name:", font_name)

    return font_prop


VIET_FONT = setup_vietnamese_font()


# ==========================================
# 3. LABEL CONFIG
# ==========================================
LABEL_MAP = {
    "ADDR": 0,
    "ADDR_PREFIX": 1,
    "AMOUNT": 2,
    "AMOUNT_PREFIX": 3,
    "BILLID": 4,
    "BILLID_PREFIX": 5,
    "CASHIER": 6,
    "CASHIER_PREFIX": 7,
    "DATETIME": 8,
    "DATETIME_PREFIX": 9,
    "FPRICE": 10,
    "FPRICE_PREFIX": 11,
    "OTHER": 12,
    "PHONE": 13,
    "PHONE_PREFIX": 14,
    "PRODUCT_NAME": 15,
    "PRODUCT_NAME_PREFIX": 16,
    "RECEMONEY": 17,
    "RECEMONEY_PREFIX": 18,
    "REMAMONEY": 19,
    "REMAMONEY_PREFIX": 20,
    "SHOP_NAME": 21,
    "SUB_TPRICE": 22,
    "SUB_TPRICE_PREFIX": 23,
    "TAMOUNT": 24,
    "TAMOUNT_PREFIX": 25,
    "TDISCOUNT": 26,
    "TDISCOUNT_PREFIX": 27,
    "TITLE": 28,
    "TPRICE": 29,
    "TPRICE_PREFIX": 30,
    "UDISCOUNT": 31,
    "UDISCOUNT_PREFIX": 32,
    "UNIT": 33,
    "UNIT_PREFIX": 34,
    "UPRICE": 35,
    "UPRICE_PREFIX": 36,
}

FALLBACK_ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

DISPLAY_ORDER = [
    "SHOP_NAME",
    "ADDR",
    "PHONE",
    "BILLID",
    "DATETIME",
    "CASHIER",
    "TITLE",
    "PRODUCT_NAME",
    "PRODUCT_NAME_PREFIX",
    "AMOUNT",
    "UNIT",
    "UPRICE",
    "FPRICE",
    "SUB_TPRICE",
    "TAMOUNT",
    "TDISCOUNT",
    "UDISCOUNT",
    "TPRICE",
    "RECEMONEY",
    "REMAMONEY",
]

FIELD_GROUPS = {
    "Store information": [
        "SHOP_NAME",
        "ADDR",
        "PHONE",
    ],
    "Receipt metadata": [
        "BILLID",
        "DATETIME",
        "CASHIER",
        "TITLE",
    ],
    "Product information": [
        "PRODUCT_NAME",
        "PRODUCT_NAME_PREFIX",
        "AMOUNT",
        "UNIT",
        "UPRICE",
        "FPRICE",
        "TPRICE",
    ],
    "Payment summary": [
        "SUB_TPRICE",
        "TAMOUNT",
        "TDISCOUNT",
        "UDISCOUNT",
        "RECEMONEY",
        "REMAMONEY",
    ],
}

LABEL_COLORS = {
    "SHOP_NAME": "#1f77b4",
    "ADDR": "#17becf",
    "PHONE": "#2ca02c",
    "BILLID": "#9467bd",
    "DATETIME": "#8c564b",
    "CASHIER": "#e377c2",
    "TITLE": "#7f7f7f",

    "PRODUCT_NAME": "#ff7f0e",
    "PRODUCT_NAME_PREFIX": "#ffbb78",
    "AMOUNT": "#bcbd22",
    "UNIT": "#7f7f7f",
    "UPRICE": "#d62728",
    "FPRICE": "#ff9896",
    "TPRICE": "#c49c94",

    "SUB_TPRICE": "#e377c2",
    "TAMOUNT": "#d62728",
    "TDISCOUNT": "#9467bd",
    "UDISCOUNT": "#8c564b",
    "RECEMONEY": "#2ca02c",
    "REMAMONEY": "#ff7f0e",
}


def get_color(label):
    if label.endswith("_PREFIX"):
        base_label = label.replace("_PREFIX", "")
        return LABEL_COLORS.get(base_label, "#333333")
    return LABEL_COLORS.get(label, "#333333")


# ==========================================
# 4. UTILS
# ==========================================
def clamp_bbox(box):
    """
    LayoutLMv3 yêu cầu bbox nằm trong khoảng [0, 1000].
    """
    return [max(0, min(1000, int(v))) for v in box]


def normalize_bbox_pixel_to_1000(box, image_width, image_height):
    """
    Convert bbox pixel [x1, y1, x2, y2] sang bbox chuẩn LayoutLM [0, 1000].
    """
    x1, y1, x2, y2 = box

    return clamp_bbox([
        1000 * x1 / image_width,
        1000 * y1 / image_height,
        1000 * x2 / image_width,
        1000 * y2 / image_height,
    ])


def get_id2label_from_model(model):
    """
    Lấy id2label từ config model.
    Nếu config bị dạng LABEL_0, LABEL_1 thì fallback về LABEL_MAP tự khai báo.
    """
    raw_id2label = model.config.id2label

    if raw_id2label is None or len(raw_id2label) == 0:
        return FALLBACK_ID2LABEL

    id2label = {}

    for k, v in raw_id2label.items():
        try:
            id2label[int(k)] = v
        except Exception:
            id2label[k] = v

    sample_values = list(id2label.values())

    if len(sample_values) > 0 and str(sample_values[0]).startswith("LABEL_"):
        return FALLBACK_ID2LABEL

    return id2label


def choose_target_image():
    if TARGET_IMAGE.strip() != "":
        return TARGET_IMAGE

    all_images = [
        f for f in os.listdir(IMG_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(all_images) == 0:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong folder: {IMG_FOLDER}")

    target_img = random.choice(all_images)
    print(f"Chọn ngẫu nhiên ảnh: {target_img}")
    return target_img


def safe_join_text(texts):
    texts = [str(t).strip() for t in texts if str(t).strip()]
    return " ".join(texts)


def shorten_text(text, max_len):
    text = str(text)

    if len(text) <= max_len:
        return text

    return text[:max_len - 3] + "..."


def safe_filename(name):
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ["-", "_", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


# ==========================================
# 5. LOAD MODEL
# ==========================================
def load_model_and_processor():
    print("Đang load LayoutLMv3 processor...")
    processor = LayoutLMv3Processor.from_pretrained(
        MODEL_DIR,
        apply_ocr=False
    )

    print("Đang load LayoutLMv3 model...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_DIR
    ).to(DEVICE)

    model.eval()

    print("Load model xong.")
    print("Device:", DEVICE)
    print("Num labels:", model.config.num_labels)

    return model, processor


# ==========================================
# 6. PREDICT
# ==========================================
def predict_one_receipt(model, processor, image_pil, tokens, norm_bboxes):
    """
    Input:
        image_pil: ảnh PIL RGB
        tokens: list text OCR
        norm_bboxes: list bbox đã normalize về [0, 1000]

    Output:
        final_preds: prediction id cho từng OCR token
        final_scores: confidence cho từng OCR token
    """

    encoding = processor(
        image_pil,
        tokens,
        boxes=norm_bboxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    try:
        word_ids = encoding.word_ids(batch_index=0)
    except Exception:
        word_ids = encoding.encodings[0].word_ids

    encoding = encoding.to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    word_pred_ids = {}
    word_scores = {}

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        # Chỉ lấy subword đầu tiên của mỗi OCR token
        if word_id not in word_pred_ids:
            pred_id = int(pred_ids[token_idx])
            score = float(probs[token_idx][pred_id])

            word_pred_ids[word_id] = pred_id
            word_scores[word_id] = score

    final_preds = []
    final_scores = []

    other_id = LABEL_MAP.get("OTHER", 12)

    for i in range(len(tokens)):
        final_preds.append(word_pred_ids.get(i, other_id))
        final_scores.append(word_scores.get(i, 0.0))

    return final_preds, final_scores


# ==========================================
# 7. EXTRACT STRUCTURED VALUES
# ==========================================
def build_extracted_result(tokens, pixel_bboxes, pred_ids, scores, id2label):
    extracted = {}
    prefixes = {}
    extracted_scores = {}
    prefix_scores = {}

    for txt, box, pred_id, score in zip(tokens, pixel_bboxes, pred_ids, scores):
        label = id2label.get(int(pred_id), "OTHER")
        txt = str(txt).strip()

        if not txt:
            continue

        if label == "OTHER":
            continue

        if score < CONF_THRESHOLD:
            continue

        if label.endswith("_PREFIX"):
            base_label = label.replace("_PREFIX", "")
            prefixes.setdefault(base_label, []).append(txt)
            prefix_scores.setdefault(base_label, []).append(score)
        else:
            extracted.setdefault(label, []).append(txt)
            extracted_scores.setdefault(label, []).append(score)

    return extracted, prefixes, extracted_scores, prefix_scores


def get_average_score(scores):
    if scores is None or len(scores) == 0:
        return None

    return float(np.mean(scores))


# ==========================================
# 8. VISUALIZATION
# ==========================================
def apply_font_kwargs(kwargs):
    """
    Gắn font tiếng Việt cho matplotlib text.
    """
    if VIET_FONT is not None:
        kwargs["fontproperties"] = VIET_FONT
    return kwargs


def draw_prediction_boxes(ax, img_rgb, tokens, pixel_bboxes, pred_ids, scores, id2label):
    ax.imshow(img_rgb)

    title_kwargs = dict(
        label="LayoutLMv3 Prediction",
        color="black" if REPORT_MODE else "#E0E0E0",
        fontsize=20,
        fontweight="bold",
        pad=16
    )
    if VIET_FONT is not None:
        title_kwargs["fontproperties"] = VIET_FONT

    ax.set_title(**title_kwargs)
    ax.axis("off")

    for box, pred_id, score, txt in zip(pixel_bboxes, pred_ids, scores, tokens):
        label = id2label.get(int(pred_id), "OTHER")
        txt = str(txt).strip()

        if not txt:
            continue

        if label == "OTHER":
            continue

        if score < CONF_THRESHOLD:
            continue

        if HIDE_PREFIX_ON_IMAGE and label.endswith("_PREFIX"):
            continue

        color = get_color(label)
        x1, y1, x2, y2 = box

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        rect_bg = mpatches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=0,
            facecolor=color,
            alpha=0.20 if REPORT_MODE else 0.30
        )
        ax.add_patch(rect_bg)

        rect_border = mpatches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect_border)

        if SHOW_CONF_ON_BOX:
            display_label = f"{label} {score:.2f}"
        else:
            display_label = label

        text_kwargs = dict(
            x=x1,
            y=max(0, y1 - 8),
            s=display_label,
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(
                facecolor=color,
                alpha=0.95,
                boxstyle="round,pad=0.25",
                edgecolor="none"
            )
        )
        text_kwargs = apply_font_kwargs(text_kwargs)

        ax.text(**text_kwargs)


def build_table_rows(extracted, prefixes, extracted_scores, prefix_scores):
    rows = []
    used_labels = set()

    for group_name, labels in FIELD_GROUPS.items():
        group_has_content = any(
            label in extracted or label in prefixes
            for label in labels
        )

        if not group_has_content:
            continue

        rows.append([f"[{group_name}]", "", "", ""])

        for label in labels:
            if label not in extracted and label not in prefixes:
                continue

            used_labels.add(label)

            prefix = safe_join_text(prefixes.get(label, []))
            value = safe_join_text(extracted.get(label, []))

            if not prefix:
                prefix = "-"

            if not value:
                value = "-"

            avg_score = get_average_score(extracted_scores.get(label, []))

            if avg_score is None:
                avg_score = get_average_score(prefix_scores.get(label, []))

            if avg_score is None:
                conf_text = "-"
            else:
                conf_text = f"{avg_score:.2f}"

            prefix = shorten_text(prefix, 24)
            value = shorten_text(value, 42)

            rows.append([label, prefix, value, conf_text])

    remaining_labels = sorted(
        set(list(extracted.keys()) + list(prefixes.keys())) - used_labels
    )

    if len(remaining_labels) > 0:
        rows.append(["[Other fields]", "", "", ""])

        for label in remaining_labels:
            prefix = safe_join_text(prefixes.get(label, []))
            value = safe_join_text(extracted.get(label, []))

            if not prefix:
                prefix = "-"

            if not value:
                value = "-"

            avg_score = get_average_score(extracted_scores.get(label, []))

            if avg_score is None:
                avg_score = get_average_score(prefix_scores.get(label, []))

            if avg_score is None:
                conf_text = "-"
            else:
                conf_text = f"{avg_score:.2f}"

            prefix = shorten_text(prefix, 24)
            value = shorten_text(value, 42)

            rows.append([label, prefix, value, conf_text])

    return rows


def draw_extracted_table(ax, extracted, prefixes, extracted_scores, prefix_scores):
    """
    Bản mới: dùng matplotlib table thay vì text monospace.
    Cách này render tiếng Việt ổn hơn và đẹp hơn cho report.
    """
    ax.set_facecolor("white" if REPORT_MODE else "#1E1E1E")
    ax.axis("off")

    title_kwargs = dict(
        label="Structured Extracted Values",
        color="black" if REPORT_MODE else "#E0E0E0",
        fontsize=20,
        fontweight="bold",
        pad=16
    )
    if VIET_FONT is not None:
        title_kwargs["fontproperties"] = VIET_FONT

    ax.set_title(**title_kwargs)

    rows = build_table_rows(
        extracted,
        prefixes,
        extracted_scores,
        prefix_scores
    )

    if len(rows) == 0:
        text_kwargs = dict(
            x=0.02,
            y=0.95,
            s="No extracted fields above confidence threshold.",
            transform=ax.transAxes,
            color="#222222" if REPORT_MODE else "#00FFCC",
            fontsize=12,
            verticalalignment="top"
        )
        text_kwargs = apply_font_kwargs(text_kwargs)
        ax.text(**text_kwargs)
        return

    col_labels = ["FIELD", "PREFIX", "VALUE", "CONF."]
    col_widths = [0.23, 0.25, 0.42, 0.10]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="left",
        loc="upper left",
        bbox=[0.00, 0.00, 1.00, 0.95]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.35)

    # Style header
    for col in range(len(col_labels)):
        cell = table[0, col]
        cell.set_facecolor("#222222")
        cell.set_text_props(color="white", weight="bold")
        if VIET_FONT is not None:
            cell.get_text().set_fontproperties(VIET_FONT)

    # Style body
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.6)

        if VIET_FONT is not None:
            cell.get_text().set_fontproperties(VIET_FONT)

        if row_idx == 0:
            continue

        text = cell.get_text().get_text()

        # Group row
        if col_idx == 0 and text.startswith("[") and text.endswith("]"):
            for c in range(len(col_labels)):
                group_cell = table[row_idx, c]
                group_cell.set_facecolor("#EEEEEE")
                group_cell.set_text_props(weight="bold", color="#111111")
                if VIET_FONT is not None:
                    group_cell.get_text().set_fontproperties(VIET_FONT)
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color="#222222")


def save_original_vs_prediction(img_rgb, tokens, pixel_bboxes, pred_ids, scores, id2label, base_name):
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16, 10),
        facecolor="white" if REPORT_MODE else "#1E1E1E"
    )

    ax1.imshow(img_rgb)

    title_kwargs = dict(
        label="Original Receipt",
        color="black" if REPORT_MODE else "#E0E0E0",
        fontsize=18,
        fontweight="bold",
        pad=14
    )
    if VIET_FONT is not None:
        title_kwargs["fontproperties"] = VIET_FONT

    ax1.set_title(**title_kwargs)
    ax1.axis("off")

    draw_prediction_boxes(
        ax2,
        img_rgb,
        tokens,
        pixel_bboxes,
        pred_ids,
        scores,
        id2label
    )

    plt.tight_layout()

    safe_base = safe_filename(base_name)

    png_path = os.path.join(SAVE_FOLDER, f"{safe_base}_original_vs_prediction.png")
    plt.savefig(
        png_path,
        dpi=SAVE_DPI,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )

    if SAVE_PDF:
        pdf_path = os.path.join(SAVE_FOLDER, f"{safe_base}_original_vs_prediction.pdf")
        plt.savefig(
            pdf_path,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight"
        )

    plt.close(fig)
    print("Saved:", png_path)


def save_full_dashboard(
    img_rgb,
    tokens,
    pixel_bboxes,
    pred_ids,
    scores,
    id2label,
    extracted,
    prefixes,
    extracted_scores,
    prefix_scores,
    base_name
):
    """
    Dashboard report:
    Chỉ gồm LayoutLMv3 Prediction + Structured Extracted Values.
    Không hiển thị ảnh original để hình gọn và tập trung hơn.
    """

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(22, 12),
        facecolor="white" if REPORT_MODE else "#1E1E1E",
        gridspec_kw={
            "width_ratios": [1.05, 1.25]
        }
    )

    # Cột 1: ảnh đã vẽ prediction box
    draw_prediction_boxes(
        ax1,
        img_rgb,
        tokens,
        pixel_bboxes,
        pred_ids,
        scores,
        id2label
    )

    # Cột 2: bảng extracted values
    draw_extracted_table(
        ax2,
        extracted,
        prefixes,
        extracted_scores,
        prefix_scores
    )

    plt.tight_layout()

    safe_base = safe_filename(base_name)

    png_path = os.path.join(SAVE_FOLDER, f"{safe_base}_layoutlmv3_extraction_dashboard.png")
    plt.savefig(
        png_path,
        dpi=SAVE_DPI,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )

    if SAVE_PDF:
        pdf_path = os.path.join(SAVE_FOLDER, f"{safe_base}_layoutlmv3_extraction_dashboard.pdf")
        plt.savefig(
            pdf_path,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight"
        )

    print("Saved:", png_path)
    plt.show()

# ==========================================
# 9. MAIN
# ==========================================
def run_layoutlmv3_report_visual():
    model, processor = load_model_and_processor()
    id2label = get_id2label_from_model(model)

    target_img = choose_target_image()
    target_json = target_img.rsplit(".", 1)[0] + ".json"

    img_path = os.path.join(IMG_FOLDER, target_img)
    json_path = os.path.join(JSON_FOLDER, target_json)

    if not os.path.exists(img_path):
        print(f"Không tìm thấy ảnh: {img_path}")
        return

    if not os.path.exists(json_path):
        print(f"Không tìm thấy JSON: {json_path}")
        return

    print("Ảnh:", img_path)
    print("JSON:", json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "tokens" not in data:
        print("JSON không có key 'tokens'.")
        return

    if "bboxes" not in data:
        print("JSON không có key 'bboxes'.")
        return

    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print(f"Không đọc được ảnh: {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_height, image_width = img_rgb.shape[:2]

    image_pil = Image.fromarray(img_rgb).convert("RGB")

    tokens = [str(t).strip() for t in data["tokens"]]
    pixel_bboxes = data["bboxes"]

    if "bboxes_norm" in data:
        norm_bboxes = [clamp_bbox(box) for box in data["bboxes_norm"]]
    else:
        norm_bboxes = [
            normalize_bbox_pixel_to_1000(box, image_width, image_height)
            for box in pixel_bboxes
        ]

    if len(tokens) != len(pixel_bboxes):
        print("Số lượng tokens và bboxes không khớp.")
        print("len(tokens):", len(tokens))
        print("len(bboxes):", len(pixel_bboxes))
        return

    if len(tokens) != len(norm_bboxes):
        print("Số lượng tokens và bboxes_norm không khớp.")
        print("len(tokens):", len(tokens))
        print("len(norm_bboxes):", len(norm_bboxes))
        return

    print("Số OCR tokens:", len(tokens))

    pred_ids, scores = predict_one_receipt(
        model=model,
        processor=processor,
        image_pil=image_pil,
        tokens=tokens,
        norm_bboxes=norm_bboxes
    )

    print("\n===== RAW PREDICTIONS =====")
    for token, pred_id, score in zip(tokens, pred_ids, scores):
        label = id2label.get(int(pred_id), "OTHER")
        print(f"{token:<35} => {label:<20} {score:.4f}")

    extracted, prefixes, extracted_scores, prefix_scores = build_extracted_result(
        tokens=tokens,
        pixel_bboxes=pixel_bboxes,
        pred_ids=pred_ids,
        scores=scores,
        id2label=id2label
    )

    base_name = target_img.rsplit(".", 1)[0]

    save_original_vs_prediction(
        img_rgb=img_rgb,
        tokens=tokens,
        pixel_bboxes=pixel_bboxes,
        pred_ids=pred_ids,
        scores=scores,
        id2label=id2label,
        base_name=base_name
    )

    save_full_dashboard(
        img_rgb=img_rgb,
        tokens=tokens,
        pixel_bboxes=pixel_bboxes,
        pred_ids=pred_ids,
        scores=scores,
        id2label=id2label,
        extracted=extracted,
        prefixes=prefixes,
        extracted_scores=extracted_scores,
        prefix_scores=prefix_scores,
        base_name=base_name
    )


if __name__ == "__main__":
    run_layoutlmv3_report_visual()