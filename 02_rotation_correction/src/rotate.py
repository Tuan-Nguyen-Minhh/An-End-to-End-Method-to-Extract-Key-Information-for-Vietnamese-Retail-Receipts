import os
import csv
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms, models


def build_resnet18_scratch(num_classes=4):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path, device, num_classes=4):
    model = build_resnet18_scratch(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def is_image_file(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def softmax_probs(logits):
    return torch.softmax(logits, dim=1)


def classify_single_image(model, pil_img, transform, device):
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = softmax_probs(logits)[0].detach().cpu()
    return probs


def evaluate_four_rotations(model, img, transform, device, class0_index=0):
    candidates = {
        0: img,
        90: img.rotate(90, expand=True),
        180: img.rotate(180, expand=True),
        270: img.rotate(270, expand=True),
    }

    results = {}

    for angle, candidate_img in candidates.items():
        probs = classify_single_image(model, candidate_img, transform, device)
        p0 = float(probs[class0_index].item())
        pred_idx = int(torch.argmax(probs).item())
        results[angle] = {
            "image": candidate_img,
            "probs": probs.tolist(),
            "p0": p0,
            "pred_idx": pred_idx,
        }

    best_angle = max(results.keys(), key=lambda a: results[a]["p0"])
    best_info = results[best_angle]
    return best_angle, best_info, results


def decide_status(best_p0, accept_thres=0.90, review_thres=0.60):
    if best_p0 >= accept_thres:
        return "accepted"
    elif best_p0 >= review_thres:
        return "review"
    else:
        return "suspect"


def process_folder(
    model,
    input_dir,
    output_dir,
    img_size,
    device,
    class_names,
    class0_index,
    accept_thres,
    review_thres,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    accepted_dir = output_dir / "images"
    review_dir = output_dir / "review"
    suspect_dir = output_dir / "suspect"

    accepted_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    suspect_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "rotation_log.csv"
    transform = get_transform(img_size)

    image_paths = [p for p in input_dir.rglob("*") if p.is_file() and is_image_file(p)]
    image_paths = sorted(image_paths)

    print(f"Found {len(image_paths)} images")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name",
            "relative_path",
            "best_rotation_angle",
            "best_p0",
            "status",
            "p0_rot0",
            "p0_rot90",
            "p0_rot180",
            "p0_rot270",
            "pred_class_rot0",
            "pred_class_rot90",
            "pred_class_rot180",
            "pred_class_rot270",
        ])

        for idx, img_path in enumerate(image_paths, 1):
            try:
                img = Image.open(img_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                print(f"[Skip] Cannot open {img_path}: {e}")
                continue

            best_angle, best_info, results = evaluate_four_rotations(
                model=model,
                img=img,
                transform=transform,
                device=device,
                class0_index=class0_index,
            )

            best_p0 = best_info["p0"]
            status = decide_status(best_p0, accept_thres, review_thres)

            if status == "accepted":
                save_root = accepted_dir
            elif status == "review":
                save_root = review_dir
            else:
                save_root = suspect_dir

            rel_path = img_path.relative_to(input_dir)
            save_path = save_root / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            best_info["image"].save(save_path)

            pred_class_rot0 = class_names[results[0]["pred_idx"]]
            pred_class_rot90 = class_names[results[90]["pred_idx"]]
            pred_class_rot180 = class_names[results[180]["pred_idx"]]
            pred_class_rot270 = class_names[results[270]["pred_idx"]]

            writer.writerow([
                img_path.name,
                str(rel_path).replace("\\", "/"),
                best_angle,
                round(best_p0, 6),
                status,
                round(results[0]["p0"], 6),
                round(results[90]["p0"], 6),
                round(results[180]["p0"], 6),
                round(results[270]["p0"], 6),
                pred_class_rot0,
                pred_class_rot90,
                pred_class_rot180,
                pred_class_rot270,
            ])

            if idx % 50 == 0 or idx == len(image_paths):
                print(f"[{idx}/{len(image_paths)}] done")

    print(f"\nSaved rotated images to: {output_dir}")
    print(f"CSV log saved to: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="Path to best .pt model")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing crop_black_mask images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save rotated results")

    parser.add_argument("--img_size", type=int, default=224, help="Input image size used in training")
    parser.add_argument("--accept_thres", type=float, default=0.90, help="Accepted threshold for best P(class0)")
    parser.add_argument("--review_thres", type=float, default=0.60, help="Review threshold for best P(class0)")

    parser.add_argument(
        "--class_names",
        type=str,
        nargs=4,
        default=["0", "90", "180", "270"],
        help='Order of classes in training, e.g. --class_names 0 90 180 270'
    )

    parser.add_argument(
        "--class0_label",
        type=str,
        default="0",
        help='Which label means upright orientation, usually "0"'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    class_names = args.class_names
    if args.class0_label not in class_names:
        raise ValueError(f'class0_label="{args.class0_label}" not found in class_names={class_names}')

    class0_index = class_names.index(args.class0_label)

    model = load_model(
        model_path=args.model_path,
        device=device,
        num_classes=len(class_names),
    )

    process_folder(
        model=model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        device=device,
        class_names=class_names,
        class0_index=class0_index,
        accept_thres=args.accept_thres,
        review_thres=args.review_thres,
    )


if __name__ == "__main__":
    main()