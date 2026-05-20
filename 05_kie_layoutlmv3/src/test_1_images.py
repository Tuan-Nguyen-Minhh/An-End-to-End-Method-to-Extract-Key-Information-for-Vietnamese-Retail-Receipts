# ==========================================
# IMPORTS
# ==========================================
import os
import json
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
from matplotlib import font_manager
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification
)


# ==========================================
# 1. CONFIG
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(base_dir, "../../models/layoutlmv3_best_model")
INPUT_IMAGE_PATH = os.path.join(base_dir, "../input_ocr_text/000595_jpg.rf.db17d6c6020b439b4368a070afc28d1f.jpg")
INPUT_JSON_PATH = os.path.join(base_dir, "../input_ocr_text/000595_jpg.rf.db17d6c6020b439b4368a070afc28d1f.json")
SAVE_FOLDER = os.path.join(base_dir, "../output_final_extraction")

os.makedirs(SAVE_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHOW_CONF_ON_BOX = False
CONF_THRESHOLD = 0.50
SAVE_PDF = True
SAVE_DPI = 150
HIDE_PREFIX_ON_IMAGE = True


# ==========================================
# 2. VIETNAMESE FONT
# ==========================================
def setup_vietnamese_font():
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
        plt.rcParams["pdf.fonttype"] = 42
        return None

    font_manager.fontManager.addfont(selected_font_path)
    font_prop = font_manager.FontProperties(fname=selected_font_path)

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["pdf.fonttype"] = 42

    return font_prop

VIET_FONT = setup_vietnamese_font()


# ==========================================
# 3. LABEL CONFIG
# ==========================================
LABEL_MAP = {
    "ADDR": 0, "ADDR_PREFIX": 1, "AMOUNT": 2, "AMOUNT_PREFIX": 3,
    "BILLID": 4, "BILLID_PREFIX": 5, "CASHIER": 6, "CASHIER_PREFIX": 7,
    "DATETIME": 8, "DATETIME_PREFIX": 9, "FPRICE": 10, "FPRICE_PREFIX": 11,
    "OTHER": 12, "PHONE": 13, "PHONE_PREFIX": 14, "PRODUCT_NAME": 15,
    "PRODUCT_NAME_PREFIX": 16, "RECEMONEY": 17, "RECEMONEY_PREFIX": 18,
    "REMAMONEY": 19, "REMAMONEY_PREFIX": 20, "SHOP_NAME": 21,
    "SUB_TPRICE": 22, "SUB_TPRICE_PREFIX": 23, "TAMOUNT": 24,
    "TAMOUNT_PREFIX": 25, "TDISCOUNT": 26, "TDISCOUNT_PREFIX": 27,
    "TITLE": 28, "TPRICE": 29, "TPRICE_PREFIX": 30, "UDISCOUNT": 31,
    "UDISCOUNT_PREFIX": 32, "UNIT": 33, "UNIT_PREFIX": 34, "UPRICE": 35,
    "UPRICE_PREFIX": 36,
}

FALLBACK_ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

LABEL_DISPLAY = {
    "SHOP_NAME": "Shop", "ADDR": "Address", "PHONE": "Phone",
    "BILLID": "Bill ID", "DATETIME": "Date", "CASHIER": "Cashier",
    "TITLE": "Title", "PRODUCT_NAME": "Product", "AMOUNT": "Qty",
    "UNIT": "Unit", "UPRICE": "Unit Price", "FPRICE": "Final Price",
    "TPRICE": "Total", "SUB_TPRICE": "Subtotal", "TAMOUNT": "T.Amount",
    "TDISCOUNT": "Discount", "UDISCOUNT": "U.Discount",
    "RECEMONEY": "Received", "REMAMONEY": "Change",
}

LABEL_COLORS = {
    "SHOP_NAME": "#1f77b4", "ADDR": "#17becf", "PHONE": "#2ca02c",
    "BILLID": "#9467bd", "DATETIME": "#8c564b", "CASHIER": "#e377c2",
    "TITLE": "#7f7f7f", "PRODUCT_NAME": "#ff7f0e", "AMOUNT": "#bcbd22",
    "UNIT": "#7f7f7f", "UPRICE": "#d62728", "FPRICE": "#ff9896",
    "TPRICE": "#c49c94", "SUB_TPRICE": "#e377c2", "TAMOUNT": "#d62728",
    "TDISCOUNT": "#9467bd", "UDISCOUNT": "#8c564b", "RECEMONEY": "#2ca02c",
    "REMAMONEY": "#ff7f0e",
}

def get_color(label):
    if label.endswith("_PREFIX"):
        label = label.replace("_PREFIX", "")
    return LABEL_COLORS.get(label, "#333333")


# ==========================================
# 4. UTILS
# ==========================================
def apply_font_kwargs(kwargs):
    if VIET_FONT:
        kwargs["fontproperties"] = VIET_FONT
    return kwargs

def clamp_bbox(box):
    return [max(0, min(1000, int(v))) for v in box]

def normalize_bbox_pixel_to_1000(box, image_width, image_height):
    x1, y1, x2, y2 = box
    return clamp_bbox([
        1000 * x1 / image_width,
        1000 * y1 / image_height,
        1000 * x2 / image_width,
        1000 * y2 / image_height,
    ])

def safe_join_text(texts):
    return " ".join([str(t).strip() for t in texts if str(t).strip()])

def safe_filename(name):
    return "".join([ch if ch.isalnum() or ch in ["-", "_", "."] else "_" for ch in str(name)])

def shortening_text(text, max_len):
    t = str(text)
    return (t[:max_len - 3] + "..." if len(t) > max_len else t)


# ==========================================
# 5. MODEL
# ==========================================
def load_model_and_processor():
    print("Loading LayoutLMv3...")
    processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    return model, processor

def get_id2label_from_model(model):
    raw_id2label = model.config.id2label
    if raw_id2label is None or len(raw_id2label) == 0:
        return FALLBACK_ID2LABEL

    id2label = {int(k): v for k, v in raw_id2label.items() if not str(v).isnumeric()}
    if not id2label or list(id2label.values())[0].startswith("LABEL_"):
        return FALLBACK_ID2LABEL
    return id2label

def predict_one_receipt(model, processor, image_pil, tokens, norm_bboxes):
    encoding = processor(
        image_pil, tokens, boxes=norm_bboxes, truncation=True,
        padding="max_length", max_length=512, return_tensors="pt"
    )

    try:
        word_ids = encoding.word_ids(batch_index=0)
    except:
        word_ids = encoding.encodings[0].word_ids

    encoding = encoding.to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
    pred_ids = outputs.logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    word_pred_ids = {}
    word_scores = {}

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in word_pred_ids:
            continue
        word_pred_ids[word_id] = int(pred_ids[token_idx])
        word_scores[word_id] = float(probs[token_idx][pred_ids[token_idx]])

    other_id = LABEL_MAP.get("OTHER", 12)
    final_preds = [word_pred_ids.get(i, other_id) for i in range(len(tokens))]
    final_scores = [word_scores.get(i, 0.0) for i in range(len(tokens))]

    return final_preds, final_scores


# ==========================================
# 6. JSON EXPORT
# ==========================================
def build_extracted_json(extracted):
    result = {}
    for label, texts in extracted.items():
        val = safe_join_text(texts)
        if val:
            result[label] = val
    return result


# ==========================================
# 7. VISUALIZATION
# ==========================================
def draw_prediction_boxes(ax, img_rgb, tokens, pixel_bboxes, pred_ids, scores, id2label):
    ax.imshow(img_rgb)
    ax.axis("off")

    for box, pred_id, score, txt in zip(pixel_bboxes, pred_ids, scores, tokens):
        label = id2label.get(int(pred_id), "OTHER")
        if not str(txt).strip() or label == "OTHER" or score < CONF_THRESHOLD:
            continue
        if HIDE_PREFIX_ON_IMAGE and label.endswith("_PREFIX"):
            continue

        color = get_color(label)
        x1, y1, x2, y2 = box
        w, h = max(1, x2 - x1), max(1, y2 - y1)

        ax.add_patch(mpatches.Rectangle((x1, y1), w, h, linewidth=0, facecolor=color, alpha=0.20))
        ax.add_patch(mpatches.Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor=color, facecolor="none"))

        disp_label = f"{label} {score:.2f}" if SHOW_CONF_ON_BOX else label
        text_kwargs = apply_font_kwargs(dict(
            x=x1, y=max(4, y1 - 5), s=disp_label, color="white",
            fontsize=7, fontweight="bold", clip_on=False,
            bbox=dict(facecolor=color, alpha=0.90, boxstyle="round,pad=0.2", edgecolor="none")
        ))
        ax.text(**text_kwargs)


# ==========================================
# 7. VISUALIZATION (ĐÃ XÓA KHUNG VIỀN NGOÀI)
# ==========================================
def draw_structured_output_panel(fig, extracted, fig_h):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Tính toán tọa độ
    title_y = 1.0 - (0.3 / fig_h)
    line_y = 1.0 - (0.6 / fig_h)
    start_y = 1.0 - (0.8 / fig_h)
    dy = 0.35 / fig_h

    # In Title
    title_kwargs = apply_font_kwargs(dict(
        x=0.05, y=title_y, s="EXTRACTED INFORMATION TABLE",
        fontsize=12, fontweight="bold", color="#111111",
        va="top", transform=ax.transAxes
    ))
    ax.text(**title_kwargs)

    # Đường kẻ ngang
    ax.plot([0.05, 0.95], [line_y, line_y], linewidth=1.2, color="#dddddd", transform=ax.transAxes)

    y = start_y
    # DUYỆT TRỰC TIẾP CÁC TRƯỜNG MODEL TRÍCH XUẤT ĐƯỢC
    for label, texts in extracted.items():
        val = safe_join_text(texts)
        if not val:
            continue

        # In Label làm Key (lấy nguyên tên Label từ Model)
        key_kwargs = apply_font_kwargs(dict(
            x=0.05, y=y, s=f"{label:15}",
            fontsize=10, fontweight="bold", color="#333333",
            family="monospace", va="top", transform=ax.transAxes
        ))
        ax.text(**key_kwargs)

        # In Value
        val_kwargs = apply_font_kwargs(dict(
            x=0.35, y=y, s=shortening_text(val, 55),
            fontsize=10, color="#111111", family="monospace",
            va="top", transform=ax.transAxes
        ))
        ax.text(**val_kwargs)

        y -= dy


# ==========================================
# 8. MAIN
# ==========================================
def run_single_image_report():
    model, processor = load_model_and_processor()
    id2label = get_id2label_from_model(model)

    if not os.path.exists(INPUT_IMAGE_PATH) or not os.path.exists(INPUT_JSON_PATH):
        print("❌ File not found")
        return

    base_name = os.path.basename(INPUT_IMAGE_PATH).rsplit(".", 1)[0]
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_bgr = cv2.imread(INPUT_IMAGE_PATH)
    if img_bgr is None:
        print("❌ Cannot read image")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    image_pil = Image.fromarray(img_rgb).convert("RGB")
    tokens = [str(t).strip() for t in data["tokens"]]
    pixel_bboxes = data["bboxes"]

    if "bboxes_norm" in data:
        norm_bboxes = [clamp_bbox(box) for box in data["bboxes_norm"]]
    else:
        norm_bboxes = [normalize_bbox_pixel_to_1000(box, w, h) for box in pixel_bboxes]

    pred_ids, scores = predict_one_receipt(model, processor, image_pil, tokens, norm_bboxes)

    extracted = {}
    for txt, pred_id, score in zip(tokens, pred_ids, scores):
        label = id2label.get(int(pred_id), "OTHER")
        txt = str(txt).strip()
        if not txt or label == "OTHER" or score < CONF_THRESHOLD or label.endswith("_PREFIX"):
            continue
        extracted.setdefault(label, []).append(txt)

    safe_base = safe_filename(base_name)

    # ======================================
    # A. PREDICTION VISUALIZATION
    # ======================================
    margin_inch = 0.15
    fig_w = w / SAVE_DPI + 2 * margin_inch
    fig_h1 = h / SAVE_DPI + 2 * margin_inch

    fig1 = plt.figure(figsize=(fig_w, fig_h1), facecolor="white")
    margin = margin_inch / fig_w
    ax1 = fig1.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    draw_prediction_boxes(ax1, img_rgb, tokens, pixel_bboxes, pred_ids, scores, id2label)

    pred_out = os.path.join(SAVE_FOLDER, f"{safe_base}_predictions.png")
    fig1.savefig(pred_out, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.05)
    if SAVE_PDF:
        fig1.savefig(pred_out.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig1)
    print(f"Saved: {pred_out}")

    # ======================================
    # B. JSON EXPORT
    # ======================================
    extracted_json = build_extracted_json(extracted)
    json_out = os.path.join(SAVE_FOLDER, f"{safe_base}_extracted.json")
    with open(json_out, "w", encoding="utf-8") as jf:
        json.dump(extracted_json, jf, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {json_out}")

    # ======================================
    # C. STRUCTURED OUTPUT PANEL (DYNAMIC HEIGHT)
    # ======================================
    valid_keys = [k for k, v in extracted.items() if safe_join_text(v)]
    num_items = len(valid_keys)
    
    # Chiều cao: 1.2 inch (Padding + Tiêu đề) + 0.35 inch cho mỗi dòng dữ liệu
    fig_h2 = max(2.0, 1.2 + num_items * 0.35)

    fig2 = plt.figure(figsize=(7, fig_h2), facecolor="white")
    draw_structured_output_panel(fig2, extracted, fig_h2)

    panel_out = os.path.join(SAVE_FOLDER, f"{safe_base}_structured_output.png")
    fig2.savefig(panel_out, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.15)
    if SAVE_PDF:
        fig2.savefig(panel_out.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig2)
    print(f"Saved structured output: {panel_out}")
    print("\nDone!")

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    run_single_image_report()