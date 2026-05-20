# An End-to-End Method to Extract Key Information for Vietnamese Retail Receipts

An end-to-end Computer Vision and Deep Learning pipeline to automatically segment, normalize, detect, transcribe, and extract key structured information from photos of Vietnamese retail receipts.

## Pipeline Architecture

![End-to-End Pipeline](full_pipeline.drawio.png)

## Methodology & Stages

### Stage 1: Receipt Segmentation & Detection (01_Segmentation_And_Detection_YOLO)
Automatically localizes and segments the receipt from noisy backgrounds using a YOLOv8 Segmentation model.

**Input Raw Image:**
![Raw Input](01_Segmentation_And_Detection_YOLO/sample/01_raw.jpg)

**Detection Visual:**
![Detection Visual](01_Segmentation_And_Detection_YOLO/sample/02_detection_visual.jpg)

### Stage 2: Receipt Image Normalization (02_Normalize_Receipts)
A multi-step preprocessing suite to clean up the segmented document.

**2.1 Background Removal & Cropping (01_background_removal_and_cropping):**
Masks out the background pixels to black and crops the image strictly to the receipt's bounding box.

**Segmented Mask & Cropped Result:**
![Segmented Mask](02_Normalize_Receipts/01_background_removal_and_cropping/samples/02_segmented_mask.jpg)
![Cropped Result](02_Normalize_Receipts/01_background_removal_and_cropping/samples/03_cropped_result.jpg)

**2.2 Rotation Correction (02_rotation_correction):**
Classifies the orientation of the cropped receipt and rotates it upright.

**Rotated Result:**
![Rotated Result](02_Normalize_Receipts/02_rotation_correction/samples/04_rotated_result.jpg)

**2.3 Deskewing & Enhancing (03_deskewing_and_enhancing):**
Evaluates text alignment angles and performs deskewing to align text lines horizontally.

**Deskewed Result:**
![Deskewed Result](02_Normalize_Receipts/03_deskewing_and_enhancing/samples/05_deskewed_enhanced.jpg)

### Stage 3: Text Bounding Box Detection (03_text_detection)
Identifies and locates individual text lines and blocks across the normalized, upright receipt.

**Text Detection:**
![Text Detection](03_text_detection/samples/06_text_detection.jpg)

### Stage 4: Text Recognition (OCR) (04_text_recognition)
Converts cropped text line boxes into digital text, optimized for Vietnamese character diacritics.

**Visual OCR:**
![Visual OCR](04_text_recognition/samples/visual_ocr.jpg)

### Stage 5: Key Information Extraction (05_kie_layoutlmv3)
Extracts semantic entities from the recognized text using LayoutLMv3, combining textual features, visual features, and spatial 2D coordinates.

Target Entities:
- SHOP_NAME
- ADDR
- PHONE
- PRODUCT_NAME
- AMOUNT
- UPRICE
- SUB_TPRICE
- TPRICE

## Repository Directory Layout

```bash
.
├── 01_Segmentation_And_Detection_YOLO/
├── 02_Normalize_Receipts/
│   ├── 01_background_removal_and_cropping/
│   ├── 02_rotation_correction/
│   └── 03_deskewing_and_enhancing/
├── 03_text_detection/
├── 04_text_recognition/
├── 05_kie_layoutlmv3/
├── models/
├── datasets/
├── utils/
├── README.md
└── full_pipeline.drawio.png
```

## Extracted Information Sample Output

The final result of the extraction pipeline generates a highly structured dictionary/JSON mapped as follows:

| Field Label | Extracted Text |
| :--- | :--- |
| SHOP_NAME | FICITEA & COFFEE |
| ADDR | 148 Hòn Khói Ninh Diêm Ninh Hòa, Khánh H |
| PHONE | 0852278979 |
| TITLE | HOÁ ĐƠN |
| PRODUCT_NAME | Trân châu Truyền Cookiesa Cream Sữa tươi Trân châu D... |
| AMOUNT | 2 2 2 1 1 1 |
| UPRICE | 19.000 5.000 30.000 25.000 5.000 |
| TPRICE | 108.000đ |