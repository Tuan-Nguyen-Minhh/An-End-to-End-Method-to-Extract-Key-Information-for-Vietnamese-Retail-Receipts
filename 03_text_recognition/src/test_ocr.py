import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# ==========================================
# 1. PARSE YOLO LABEL
# ==========================================
def parse_yolo_label(label_path: str, img_w: int, img_h: int) -> list:
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            x1 = max(0,     int((cx - w / 2) * img_w))
            y1 = max(0,     int((cy - h / 2) * img_h))
            x2 = min(img_w, int((cx + w / 2) * img_w))
            y2 = min(img_h, int((cy + h / 2) * img_h))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return boxes

# ==========================================
# 2. VISUAL HELPERS
# ==========================================
FONT_SIZE  = 13
BOX_THICK  = 1
LABEL_PAD  = 3
HEADER_H   = 28

# Colors (BGR)
CLR_OK         = (90,  130, 200)   # Blue — OCR raw
CLR_ORIGIN_BOX = (180, 180, 180)   # Light gray — original image
CLR_HEADER_ORI = (25,  25,  25)    # Dark gray — original
CLR_HEADER_RAW = (30,  30,  30)    # Dark — raw

def _get_font(size: int = FONT_SIZE):
    for name in ["DejaVuSans.ttf", "arial.ttf", "FreeSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except IOError:
            pass
    return ImageFont.load_default()

def draw_panel(cv2_img: np.ndarray, annotations: list) -> np.ndarray:
    """
    Draw boxes + labels on image using PIL (supports Vietnamese text).
    annotations: list of (box, text, color_bgr)
    """
    pil  = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _get_font()

    for (x1, y1, x2, y2), text, color in annotations:
        rgb = color[::-1]   # BGR -> RGB
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=BOX_THICK)

        if not text:
            continue

        try:
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            tw, th = draw.textsize(text, font=font)

        lx1 = x1
        ly1 = max(0, y1 - th - LABEL_PAD * 2)
        lx2 = x1 + tw + LABEL_PAD * 2
        ly2 = max(0, y1)
        
        draw.rectangle([lx1, ly1, lx2, ly2], fill=rgb)
        draw.text((lx1 + LABEL_PAD, ly1 + LABEL_PAD - 1), text,
                  font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def add_header(cv2_img: np.ndarray, title: str, bg_bgr: tuple) -> np.ndarray:
    """Thin title bar above panel."""
    h_bar = np.full((HEADER_H, cv2_img.shape[1], 3), bg_bgr, dtype=np.uint8)
    pil   = Image.fromarray(cv2.cvtColor(h_bar, cv2.COLOR_BGR2RGB))
    draw  = ImageDraw.Draw(pil)
    font  = _get_font(12)
    draw.text((10, 7), title, font=font, fill=(210, 210, 210))
    bar   = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return np.vstack([bar, cv2_img])

def pad_height(im: np.ndarray, target_h: int) -> np.ndarray:
    diff = target_h - im.shape[0]
    if diff <= 0:
        return im
    return np.vstack([im, np.zeros((diff, im.shape[1], 3), dtype=np.uint8)])

# ==========================================
# 3. MAIN PIPELINE (VISUALIZE RAW OCR)
# ==========================================
def run_visual_test(image_path: str, label_path: str, vietocr_model):
    print("=" * 80)
    print(f"Processing image: {os.path.basename(image_path)}")
    print("=" * 80)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return

    img_h, img_w = img.shape[:2]
    boxes = parse_yolo_label(label_path, img_w, img_h)

    if not boxes:
        print("No boxes found in label!")
        return

    ann_ori = []   # Original image — boxes only
    ann_raw = []   # VietOCR raw (with Confidence Score)

    print(f"  {'BOX':<5} {'RAW OCR (CONF)':<40}")
    print(f"  {'-'*5} {'-'*40}")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        crop    = img[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        # Run VietOCR with probability
        raw_text, prob = vietocr_model.predict(pil_img, return_prob=True)
        conf_percent = int(prob * 100)
        raw_display  = f"{raw_text} [{conf_percent}%]"
        
        ann_ori.append((box, "",  CLR_ORIGIN_BOX))
        ann_raw.append((box, raw_display, CLR_OK))

        print(f"  {i:<5} {raw_display:<40}")

    # --- Draw 2 panels ---
    panel_ori = draw_panel(img.copy(), ann_ori)
    panel_raw = draw_panel(img.copy(), ann_raw)

    # --- Header ---
    panel_ori = add_header(panel_ori, "ORIGINAL IMAGE", CLR_HEADER_ORI)
    panel_raw = add_header(panel_raw, "RAW VIETOCR (CONF)", CLR_HEADER_RAW)

    # --- Synchronize height ---
    h = max(panel_ori.shape[0], panel_raw.shape[0])
    panel_ori = pad_height(panel_ori, h)
    panel_raw = pad_height(panel_raw, h)

    # --- Concatenate 2 panels ---
    divider = np.full((h, 2, 3), 60, dtype=np.uint8)
    final   = np.hstack([panel_ori, divider, panel_raw])

    out_path = "visual_ocr_only.jpg"
    cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("-" * 80)
    print(f"Completed recognition for {len(boxes)} boxes.")
    print(f"Comparison image saved at: {os.path.abspath(out_path)}")
    print("=" * 80)

# ==========================================
# ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    image_path   = os.path.join(base_dir, "../../01_text_detection/outputs/train/images/001019_jpg.rf.9b2caed5a2b5134fb7c28f2b2e55b244.jpg")
    label_path   = os.path.join(base_dir, "../../01_text_detection/outputs/train/labels/001019_jpg.rf.9b2caed5a2b5134fb7c28f2b2e55b244.txt")
    weights_path = os.path.join(base_dir, "../../models/vietocr_vire_receipts.pth")

    print("Loading VietOCR...")
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights']                 = weights_path
    config['cnn']['pretrained']       = False
    config['device']                  = 'cpu'
    config['predictor']['beamsearch'] = False

    try:
        vietocr_model = Predictor(config)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error: {e}")
        exit()

    run_visual_test(image_path, label_path, vietocr_model)
