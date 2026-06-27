

import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

# For VietOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# For LayoutLMv3
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# CONFIG & CONSTANTS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SEG_PATH = os.path.join(BASE_DIR, "models/bill_segmentation_best.pt")
MODEL_ROT_PATH = os.path.join(BASE_DIR, "models/resnet18_rotation_best.pt")
MODEL_DET_PATH = os.path.join(BASE_DIR, "models/text_detection_yolo_best.pt")
MODEL_OCR_PATH = os.path.join(BASE_DIR, "models/vietocr_vire_receipts.pth")
MODEL_KIE_PATH = os.path.join(BASE_DIR, "models/layoutlmv3_best_model")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detection settings
DET_CONF_THRES = 0.08
DET_IOU_THRES = 0.45
MIN_BOX_AREA = 16
EXPAND_X_RATIO = 0.01
EXPAND_Y_RATIO = 0.02
MERGE_Y_CENTER_THR = 0.30
MERGE_X_GAP_THR = 0.25
MERGE_OVERLAP_THR = 0.50

# KIE settings
KIE_CONF_THRESHOLD = 0.50

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
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
ITEM_FIELDS = {"PRODUCT_NAME", "UNIT", "AMOUNT", "UPRICE", "UDISCOUNT", "SUB_TPRICE"}.
HEADER_FOOTER_FIELDS = {
    "SHOP_NAME", "ADDR", "PHONE", "TITLE", "DATETIME", "BILLID", "CASHIER",
    "TPRICE", "TAMOUNT", "TDISCOUNT", "RECEMONEY", "REMAMONEY", "FPRICE",
}
INCLUDED_HEADER_FOOTER_FIELDS = {"SHOP_NAME", "ADDR", "DATETIME", "TPRICE"}
INCLUDED_ITEM_FIELDS = {"PRODUCT_NAME", "AMOUNT", "UPRICE", "SUB_TPRICE", "UDISCOUNT"}
# ===== END NEW =====


# MODELS LOADING
def build_resnet18_scratch(num_classes=4):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_all_models():
    models_dict = {}
    models_dict['seg'] = YOLO(MODEL_SEG_PATH)
    rot_model = build_resnet18_scratch(num_classes=4)
    checkpoint = torch.load(MODEL_ROT_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else (checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    new_state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    rot_model.load_state_dict(new_state_dict, strict=True)
    rot_model.to(DEVICE)
    rot_model.eval()
    models_dict['rot'] = rot_model

    # 3. Detection
    models_dict['det'] = YOLO(MODEL_DET_PATH)

    # 4. OCR
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = MODEL_OCR_PATH
    config['cnn']['pretrained'] = False
    config['device'] = 'cpu' if DEVICE.type == 'cpu' else 'cuda:0'
    config['predictor']['beamsearch'] = False
    models_dict['ocr'] = Predictor(config)

    # 5. KIE
    processor = LayoutLMv3Processor.from_pretrained(MODEL_KIE_PATH, apply_ocr=False)
    kie_model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_KIE_PATH).to(DEVICE)
    kie_model.eval()
    models_dict['kie_processor'] = processor
    models_dict['kie_model'] = kie_model

    return models_dict

# STAGE 1: SEGMENTATION
def stage1_segment(model, img_bgr):
    result = model.predict(img_bgr, conf=0.5, verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return img_bgr

    best_idx = 0
    if result.boxes is not None and len(result.boxes) > 0:
        best_idx = int(np.argmax(result.boxes.conf.cpu().numpy()))

    mask = result.masks.data[best_idx].cpu().numpy().astype(np.uint8)
    if mask.shape[:2] != img_bgr.shape[:2]:
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    masked_img = img_bgr * mask[:, :, None]

    # Find bounding box of mask to crop
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cropped_img = masked_img[y:y+h, x:x+w]
    else:
        cropped_img = masked_img

    return cropped_img

# STAGE 2: NORMALIZATION
def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def classify_rotation(model, pil_img):
    transform = get_transform(224)
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    return probs

def evaluate_four_rotations(model, img_pil):
    candidates = {
        0: img_pil,
        90: img_pil.rotate(90, expand=True),
        180: img_pil.rotate(180, expand=True),
        270: img_pil.rotate(270, expand=True),
    }

    results = {}
    for angle, candidate_img in candidates.items():
        probs = classify_rotation(model, candidate_img)
        p0 = float(probs[0].item())  # class0 is upright
        results[angle] = {"image": candidate_img, "p0": p0}

    best_angle = max(results.keys(), key=lambda a: results[a]["p0"])
    return results[best_angle]["image"]

def rotate_keep_canvas(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

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

def refine_angle_by_projection(bgr: np.ndarray, coarse_angle: float, search_range: float = 2.0, step: float = 0.2):
    n_steps = int(round(2 * search_range / step)) + 1
    angles = np.linspace(coarse_angle - search_range, coarse_angle + search_range, n_steps)
    best_angle = coarse_angle
    best_score = -1.0
    for angle in angles:
        rotated = rotate_keep_canvas(bgr, float(angle))
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        score = projection_score(gray)
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle

def stage2_normalize(rot_model, cropped_bgr):
    # 1. Rotation Correction
    pil_img = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
    rotated_pil = evaluate_four_rotations(rot_model, pil_img)
    rotated_bgr = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2BGR)

    # 2. Deskew and Enhance
    coarse_angle = estimate_angle_from_mask(rotated_bgr)
    refined_angle = refine_angle_by_projection(rotated_bgr, coarse_angle)
    deskewed_bgr = rotate_keep_canvas(rotated_bgr, refined_angle)

    gray = cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2GRAY)
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe_op.apply(gray)
    blur = cv2.GaussianBlur(clahe_img, (0, 0), 1.0)
    sharpened_gray = cv2.addWeighted(clahe_img, 1.5, blur, -0.5, 0)

    return sharpened_gray


# STAGE 3: TEXT DETECTION
def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def expand_box(box, img_w, img_h):
    x1, y1, x2, y2, conf, cls_id = box
    bw, bh = x2 - x1, y2 - y1
    nx1, ny1 = x1 - bw * EXPAND_X_RATIO, y1 - bh * EXPAND_Y_RATIO
    nx2, ny2 = x2 + bw * EXPAND_X_RATIO, y2 + bh * EXPAND_Y_RATIO
    nx1, ny1, nx2, ny2 = clip_box(nx1, ny1, nx2, ny2, img_w, img_h)
    return [nx1, ny1, nx2, ny2, conf, cls_id]

def should_merge(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    max_h = max(ay2 - ay1, by2 - by1)
    if max_h == 0: return False
    y_center_diff = abs((ay1 + ay2) / 2.0 - (by1 + by2) / 2.0)
    if y_center_diff > MERGE_Y_CENTER_THR * max_h: return False
    overlap_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    min_w = min(ax2 - ax1, bx2 - bx1)
    if min_w > 0 and overlap_x / min_w > MERGE_OVERLAP_THR: return False
    if bx1 >= ax2: x_gap = bx1 - ax2
    elif ax1 >= bx2: x_gap = ax1 - bx2
    else: x_gap = 0.0
    return x_gap <= MERGE_X_GAP_THR * max_h

def merge_two_boxes(box_a, box_b):
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2])
    y2 = max(box_a[3], box_b[3])
    conf = max(box_a[4], box_b[4])
    cls_id = box_a[5] if box_a[4] >= box_b[4] else box_b[5]
    return [x1, y1, x2, y2, conf, cls_id]

def merge_boxes_linewise(boxes):
    if len(boxes) <= 1: return boxes
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    used = [False] * len(boxes)
    merged = []
    for i in range(len(boxes)):
        if used[i]: continue
        used[i] = True
        cur = boxes[i]
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]: continue
                if should_merge(cur, boxes[j]):
                    cur = merge_two_boxes(cur, boxes[j])
                    used[j] = True
                    changed = True
        merged.append(cur)
    return sorted(merged, key=lambda b: (b[1], b[0]))

def stage3_detect_text(det_model, norm_img):
    img_bgr = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    h, w = img_bgr.shape[:2]
    result = det_model.predict(img_bgr, imgsz=1280, conf=DET_CONF_THRES, iou=DET_IOU_THRES, device=DEVICE, verbose=False)[0]

    raw_boxes = []
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = clip_box(xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], w, h)
            raw_boxes.append([x1, y1, x2, y2, float(confs[i]), int(classes[i])])

    boxes = [expand_box(b, w, h) for b in raw_boxes]
    boxes = [b for b in boxes if max(0, b[2] - b[0]) * max(0, b[3] - b[1]) >= MIN_BOX_AREA]
    boxes = merge_boxes_linewise(boxes)
    return boxes  # list of [x1, y1, x2, y2, conf, cls_id]


# STAGE 4: OCR
def stage4_ocr(ocr_model, norm_img, boxes):
    img_bgr = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    pil_imgs = []
    valid_indices = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pil_imgs.append(pil_img)
        valid_indices.append(i)

    results_texts = []
    if pil_imgs:
        results_texts = ocr_model.predict_batch(pil_imgs, return_prob=False)

    final_results = [""] * len(boxes)
    for idx_in_valid, orig_idx in enumerate(valid_indices):
        final_results[orig_idx] = results_texts[idx_in_valid]

    return final_results


# STAGE 5: KIE
def normalize_bbox_pixel_to_1000(box, image_width, image_height):
    x1, y1, x2, y2 = box[:4]
    return [
        max(0, min(1000, int(1000 * x1 / image_width))),
        max(0, min(1000, int(1000 * y1 / image_height))),
        max(0, min(1000, int(1000 * x2 / image_width))),
        max(0, min(1000, int(1000 * y2 / image_height))),
    ]

def stage5_kie(kie_model, processor, norm_img, boxes, texts):
    img_bgr = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    image_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    h, w = img_bgr.shape[:2]

    norm_bboxes = [normalize_bbox_pixel_to_1000(box, w, h) for box in boxes]

    valid_indices = [i for i, t in enumerate(texts) if str(t).strip()]
    if not valid_indices:
        return [], ["OTHER"] * len(boxes)

    valid_texts = [texts[i] for i in valid_indices]
    valid_norm_bboxes = [norm_bboxes[i] for i in valid_indices]

    encoding = processor(
        image_pil, valid_texts, boxes=valid_norm_bboxes, truncation=True,
        padding="max_length", max_length=512, return_tensors="pt"
    )

    try:
        word_ids = encoding.word_ids(batch_index=0)
    except Exception:
        word_ids = encoding.encodings[0].word_ids

    encoding = encoding.to(DEVICE)
    with torch.no_grad():
        outputs = kie_model(**encoding)

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
    box_labels = ["OTHER"] * len(boxes)

    # ===== CHANGED: flat list output, one entry per box, in reading order
    # (top-to-bottom, then left-to-right -- same order as `boxes`, which
    # merge_boxes_linewise() already sorted by (y1, x1) back in Stage 3). =====
    extracted_entries = []

    for i in range(len(boxes)):
        if i not in valid_indices:
            continue
        idx_in_valid = valid_indices.index(i)
        pred_id = word_pred_ids.get(idx_in_valid, other_id)
        score = word_scores.get(idx_in_valid, 0.0)

        label = ID2LABEL.get(pred_id, "OTHER")
        txt = texts[i].strip()

        # box_labels always reflects the raw model prediction -- used only
        # by draw_kie_boxes() for visualization, unaffected by the allowlist.
        box_labels[i] = label

        if label == "OTHER" or score < KIE_CONF_THRESHOLD or label.endswith("_PREFIX"):
            continue
        if not txt:
            continue
        is_header_footer = label in HEADER_FOOTER_FIELDS and label in INCLUDED_HEADER_FOOTER_FIELDS
        is_item_field = label in ITEM_FIELDS and label in INCLUDED_ITEM_FIELDS
        if not (is_header_footer or is_item_field):
            continue
        extracted_entries.append({label: txt})

    return extracted_entries, box_labels


# EXPORT & RUN PIPELINE
def export_to_json(extracted_data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)

import urllib.request
from PIL import ImageDraw, ImageFont
from matplotlib import font_manager

FONT_URL = "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf"
FONT_PATH = os.path.join(BASE_DIR, "Roboto-Regular.ttf")

# ===== Premium color palette for KIE labels =====
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

def setup_vietnamese_font():
    font_candidates = [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            return font_manager.FontProperties(fname=fp)
    # Fallback: try downloading Roboto
    if not os.path.exists(FONT_PATH):
        try:
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        except Exception:
            pass
    if os.path.exists(FONT_PATH):
        font_manager.fontManager.addfont(FONT_PATH)
        return font_manager.FontProperties(fname=FONT_PATH)
    return None

VIET_FONT = setup_vietnamese_font()

def _get_pil_font(size=20):
    """Load a Vietnamese-capable font at the given size."""
    if VIET_FONT:
        try:
            return ImageFont.truetype(VIET_FONT.get_file(), size)
        except Exception:
            pass
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

def get_vietnamese_font(size=14):
    if not os.path.exists(FONT_PATH):
        try:
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        except Exception:
            pass
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

def draw_ocr_boxes(img, boxes, texts):
    """Draw OCR results with semi-transparent blue boxes and Vietnamese text labels."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")
    font = _get_pil_font(size=20)

    for box, txt in zip(boxes, texts):
        txt = str(txt).strip()
        if not txt:
            continue
        x1, y1, x2, y2 = map(int, box[:4])

        # Semi-transparent fill + solid outline
        draw.rectangle([x1, y1, x2, y2], fill=(0, 150, 255, 30), outline=(0, 150, 255, 255), width=2)

        # Text label background
        try:
            tb = draw.textbbox((0, 0), txt, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            tw, th = draw.textsize(txt, font=font)

        lx1, ly1 = x1, max(0, y1 - th - 6)
        lx2, ly2 = x1 + tw + 8, max(0, y1)
        draw.rectangle([lx1, ly1, lx2, ly2], fill=(0, 150, 255, 240))
        draw.text((lx1 + 4, ly1 + 1), txt, font=font, fill=(255, 255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_detection_boxes(img, boxes):
    """Draw text detection boxes with green outlines and semi-transparent fill."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        draw.rectangle([x1, y1, x2, y2], fill=(0, 255, 0, 25), outline=(0, 200, 0, 255), width=2)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_kie_boxes(img, boxes, box_labels):
    """
    Draw KIE classification results with color-coded semi-transparent boxes
    and label tags — matching the premium style from export_full_pipeline_drawio.py.

    NOTE: intentionally NOT filtered by INCLUDED_HEADER_FOOTER_FIELDS /
    INCLUDED_ITEM_FIELDS. This keeps showing every field the model
    predicted for evaluation/visualization purposes.
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")
    font = _get_pil_font(size=20)

    for box, label in zip(boxes, box_labels):
        if label == "OTHER" or label.endswith("_PREFIX"):
            continue

        color_hex = get_color(label)
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)

        x1, y1, x2, y2 = map(int, box[:4])

        # Semi-transparent fill + solid outline
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 50), outline=(r, g, b, 255), width=2)

        # Label tag background
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)

        lx1, ly1 = x1, max(0, y1 - th - 6)
        lx2, ly2 = x1 + tw + 8, max(0, y1)
        draw.rectangle([lx1, ly1, lx2, ly2], fill=(r, g, b, 240))
        draw.text((lx1 + 4, ly1 + 1), label, font=font, fill=(255, 255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def run_pipeline(models, img_bgr):
    # 1. Segmentation
    cropped_img = stage1_segment(models['seg'], img_bgr)

    # 2. Normalization
    norm_gray = stage2_normalize(models['rot'], cropped_img)

    # 3. Detection
    boxes = stage3_detect_text(models['det'], norm_gray)

    # 4. OCR
    texts = stage4_ocr(models['ocr'], norm_gray, boxes)

    # 5. KIE
    extracted_json, box_labels = stage5_kie(models['kie_model'], models['kie_processor'], norm_gray, boxes, texts)

    # Draw visualizations
    det_img = draw_detection_boxes(cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR), boxes)
    ocr_img = draw_ocr_boxes(cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR), boxes, texts)
    kie_img = draw_kie_boxes(cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR), boxes, box_labels)

    return {
        "raw_img": img_bgr,
        "cropped": cropped_img,
        "normalized": norm_gray,
        "detection_img": det_img,
        "ocr_img": ocr_img,
        "kie_img": kie_img,
        "boxes": boxes,
        "extracted_json": extracted_json
    }
