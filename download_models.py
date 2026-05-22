import os
import gdown

# Single file models
MODEL_DRIVE_IDS = {
    "bill_segmentation_best.pt": "1QpdpmwJe7f6R5RrVxVM2ceNAhmAYTUYG",
    "resnet18_rotation_best.pt": "1v3H25CJ6WXA8v-y7iiKKt6wDbCXJamUk",
    "text_detection_yolo_best.pt": "1Qd-_eqkw4fuhZH0Vq2utMhA3heCx-q2x",
    "vietocr_vire_receipts.pth": "1efR4432O2YRVToEADVx3G_-Gr1jaqa4P",
}

# LayoutLMv3 directory files
LAYOUTLMV3_IDS = {
    "config.json": "18uC69OLGZ82rjZhBybQ3DINu7o4eGoR2",
    "model.safetensors": "1I-XWXssQ8_wfm31sSVsJFMaVxhGdncbw",
    "processor_config.json": "1voeQVSgWrR243sTddFHsTVV75FPYQS0p",
    "tokenizer_config.json": "1TcyJaVZ2A6jfeWiDS0ylVPvlABe3k0iy",
    "tokenizer.json": "1gCag8J9vOxlOdt9Au3FZ7A0lZI2UsGsu",
    "training_args.bin": "1NH9js_s8sd_jLN91WdPiSRZeiH4i1Y4r",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LAYOUTLMV3_DIR = os.path.join(MODELS_DIR, "layoutlmv3_best_model")

def download_file(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Downloading {output_path}...")
    try:
        gdown.download(url, output_path, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"Failed to download {output_path} via gdown: {e}")
        print(f"Attempting to download using alternative method for large files...")
        os.system(f'wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id={file_id}\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id={file_id}" -O {output_path} && rm -rf /tmp/cookies.txt')

def check_and_download_models():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Download single file models
    for filename, file_id in MODEL_DRIVE_IDS.items():
        output_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(output_path):
            download_file(file_id, output_path)

    # Download LayoutLMv3 directory files
    if not os.path.exists(LAYOUTLMV3_DIR):
        os.makedirs(LAYOUTLMV3_DIR)
        
    for filename, file_id in LAYOUTLMV3_IDS.items():
        output_path = os.path.join(LAYOUTLMV3_DIR, filename)
        if not os.path.exists(output_path):
            download_file(file_id, output_path)

if __name__ == "__main__":
    check_and_download_models()
