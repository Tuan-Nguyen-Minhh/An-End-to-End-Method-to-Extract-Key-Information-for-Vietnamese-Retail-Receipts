import os
import copy
import json
import random
import shutil
import csv
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# CẤU HÌNH CHUNG
# =========================================================
DATA_ROOT = "/content/raw_datasets_for_rotation_balanced_strict"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "valid")

OUTPUT_DIR = "/content/rotation_runs"
CLEAR_OLD_OUTPUT = True

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# SEED
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# OUTPUT DIR
# =========================================================
def prepare_output_dir(output_dir: str, clear_old: bool = True):
    if clear_old and os.path.exists(output_dir):
        print(f"[INFO] Xóa toàn bộ output cũ: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output dir sẵn sàng: {output_dir}")


# =========================================================
# SAVE / JSON / CSV / LATEX
# =========================================================
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def latex_escape(text):
    text = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


# =========================================================
# PARAM COUNT / ARCH TABLE
# =========================================================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def build_architecture_rows(small_cnn_model, resnet18_model, image_size, num_classes):
    cnn_total, cnn_trainable = count_parameters(small_cnn_model)
    res_total, res_trainable = count_parameters(resnet18_model)

    rows = [
        {
            "model_name": "small_cnn_16_32_64",
            "source": "from_scratch",
            "pretrained": "no",
            "input_size": f"{image_size}x{image_size}",
            "num_classes": num_classes,
            "conv_blocks": 3,
            "channels": "16-32-64",
            "dropout": 0.25,
            "backbone": "custom_cnn",
            "block_type": "conv-bn-relu-maxpool",
            "layers_per_stage": "-",
            "classifier": "AdaptiveAvgPool -> Flatten -> Dropout -> Linear",
            "total_params": cnn_total,
            "trainable_params": cnn_trainable,
            "notes": "3 conv blocks, kernel=3, padding=1",
        },
        {
            "model_name": "resnet18_scratch",
            "source": "from_scratch",
            "pretrained": "no",
            "input_size": f"{image_size}x{image_size}",
            "num_classes": num_classes,
            "conv_blocks": "-",
            "channels": "-",
            "dropout": 0.0,
            "backbone": "ResNet18",
            "block_type": "BasicBlock",
            "layers_per_stage": "[2,2,2,2]",
            "classifier": "fc -> Linear(num_classes)",
            "total_params": res_total,
            "trainable_params": res_trainable,
            "notes": "torchvision.models.resnet18(weights=None)",
        },
    ]
    return rows


def save_architecture_latex(rows, save_path):
    columns = [
        "model_name",
        "source",
        "pretrained",
        "input_size",
        "num_classes",
        "backbone",
        "block_type",
        "layers_per_stage",
        "channels",
        "dropout",
        "total_params",
        "trainable_params",
        "notes",
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lllllllllllll}\n")
        f.write("\\hline\n")
        f.write(" & ".join(columns) + " \\\\\n")
        f.write("\\hline\n")

        for row in rows:
            vals = [latex_escape(row[col]) for col in columns]
            f.write(" & ".join(vals) + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Model architecture configuration used in experiments.}\n")
        f.write("\\label{tab:model_architecture}\n")
        f.write("\\end{table}\n")


# =========================================================
# DATASET / DATALOADER
# =========================================================
def build_transforms():
    # Bài toán orientation classification:
    # không rotate, không flip
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return train_tf, val_tf


def count_dataset_per_class(dataset):
    counter = Counter()
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        counter[class_name] += 1
    return dict(counter)


def build_dataloaders():
    train_tf, val_tf = build_transforms()

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_tf)

    print("train classes:", train_dataset.classes)
    print("val classes  :", val_dataset.classes)

    assert train_dataset.classes == val_dataset.classes, (
        f"Train/val class order khác nhau:\n"
        f"train={train_dataset.classes}\n"
        f"val={val_dataset.classes}"
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    dataset_info = {
        "data_root": DATA_ROOT,
        "train_dir": TRAIN_DIR,
        "val_dir": VAL_DIR,
        "class_names": class_names,
        "class_to_idx": train_dataset.class_to_idx,
        "num_classes": num_classes,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_class_counts": count_dataset_per_class(train_dataset),
        "val_class_counts": count_dataset_per_class(val_dataset),
        "train_transforms": [
            "Resize((IMAGE_SIZE, IMAGE_SIZE))",
            "ColorJitter(brightness=0.15, contrast=0.15)",
            "ToTensor()",
            "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        ],
        "val_transforms": [
            "Resize((IMAGE_SIZE, IMAGE_SIZE))",
            "ToTensor()",
            "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        ],
    }

    return train_dataset, val_dataset, train_loader, val_loader, class_names, num_classes, dataset_info


# =========================================================
# MODEL 1: SMALL CNN
# =========================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# MODEL 2: RESNET18 SCRATCH
# =========================================================
def build_resnet18_scratch(num_classes=4):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =========================================================
# PLOT
# =========================================================
def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(history, save_path, title="Loss Curve"):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, label="Train Loss")
    ax.plot(epochs, val_loss, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curve(history, save_path, title="Accuracy / Macro F1 Curve"):
    epochs = [row["epoch"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    train_f1 = [row["train_macro_f1"] for row in history]
    val_f1 = [row["val_macro_f1"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, label="Train Acc")
    ax.plot(epochs, val_acc, label="Val Acc")
    ax.plot(epochs, train_f1, label="Train Macro F1")
    ax.plot(epochs, val_f1, label="Val Macro F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lr_curve(history, save_path, title="Learning Rate Curve"):
    epochs = [row["epoch"] for row in history]
    lrs = [row["lr"] for row in history]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, lrs, label="Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_final_compare(compare_dict, save_path):
    model_names = list(compare_dict.keys())
    val_accs = [compare_dict[name]["best_val_acc"] for name in model_names]
    val_f1s = [compare_dict[name]["best_val_macro_f1"] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, val_accs, width, label="Val Acc")
    ax.bar(x + width / 2, val_f1s, width, label="Val Macro F1")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Final Model Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds)

    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    return epoch_loss, epoch_acc, epoch_f1, cm, report


def run_training(model_name, model, train_loader, val_loader, device, class_names, model_config):
    print(f"\n========== TRAIN {model_name} ==========")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    best_f1 = -1.0
    best_state = None
    history = []

    run_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(run_dir, exist_ok=True)

    save_json(model_config, os.path.join(run_dir, "model_config.json"))

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1, val_cm, val_report = evaluate(
            model, val_loader, criterion, device, class_names
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_macro_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        print(
            f"[{model_name}] "
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

            torch.save(best_state, os.path.join(run_dir, "best_model.pt"))
            save_json(history, os.path.join(run_dir, "history.json"))
            save_json(val_cm.tolist(), os.path.join(run_dir, "best_confusion_matrix.json"))
            save_json(val_report, os.path.join(run_dir, "best_classification_report.json"))

            plot_confusion_matrix(
                val_cm,
                class_names,
                os.path.join(run_dir, "best_confusion_matrix.png"),
                title=f"{model_name} - Best Confusion Matrix",
            )
            plot_loss_curve(
                history,
                os.path.join(run_dir, "loss_curve.png"),
                title=f"{model_name} - Loss Curve",
            )
            plot_metric_curve(
                history,
                os.path.join(run_dir, "metric_curve.png"),
                title=f"{model_name} - Accuracy / Macro F1",
            )
            plot_lr_curve(
                history,
                os.path.join(run_dir, "lr_curve.png"),
                title=f"{model_name} - Learning Rate",
            )

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc, val_f1, val_cm, val_report = evaluate(
        model, val_loader, criterion, device, class_names
    )

    summary = {
        "model_name": model_name,
        "best_val_loss": float(val_loss),
        "best_val_acc": float(val_acc),
        "best_val_macro_f1": float(val_f1),
        "confusion_matrix": val_cm.tolist(),
        "classification_report": val_report,
        "model_config": model_config,
    }
    save_json(summary, os.path.join(run_dir, "summary.json"))

    plot_confusion_matrix(
        val_cm,
        class_names,
        os.path.join(run_dir, "final_confusion_matrix.png"),
        title=f"{model_name} - Final Best Confusion Matrix",
    )
    plot_loss_curve(
        history,
        os.path.join(run_dir, "final_loss_curve.png"),
        title=f"{model_name} - Final Loss Curve",
    )
    plot_metric_curve(
        history,
        os.path.join(run_dir, "final_metric_curve.png"),
        title=f"{model_name} - Final Accuracy / Macro F1",
    )
    plot_lr_curve(
        history,
        os.path.join(run_dir, "final_lr_curve.png"),
        title=f"{model_name} - Final Learning Rate",
    )

    print(f"\n[{model_name}] BEST RESULT")
    print(f"val_loss    : {val_loss:.4f}")
    print(f"val_acc     : {val_acc:.4f}")
    print(f"val_macro_f1: {val_f1:.4f}")
    print("confusion_matrix:")
    print(val_cm)

    return summary


# =========================================================
# MODEL CONFIGS
# =========================================================
def get_smallcnn_model_config(model, num_classes):
    total_params, trainable_params = count_parameters(model)
    return {
        "model_name": "small_cnn_16_32_64",
        "source": "from_scratch",
        "pretrained": False,
        "weights": None,
        "num_classes": num_classes,
        "input_size": IMAGE_SIZE,
        "feature_extractor": {
            "num_conv_blocks": 3,
            "channels": [16, 32, 64],
            "kernel_size": 3,
            "padding": 1,
            "normalization": "BatchNorm2d",
            "activation": "ReLU",
            "pooling": "MaxPool2d(2)",
        },
        "classifier": {
            "adaptive_avg_pool": True,
            "dropout": 0.25,
            "linear_out_features": num_classes,
        },
        "parameter_count": {
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
    }


def get_resnet18_model_config(model, num_classes):
    total_params, trainable_params = count_parameters(model)
    return {
        "model_name": "resnet18_scratch",
        "source": "from_scratch",
        "pretrained": False,
        "weights": None,
        "num_classes": num_classes,
        "input_size": IMAGE_SIZE,
        "backbone": "ResNet18",
        "block_type": "BasicBlock",
        "layers_per_stage": [2, 2, 2, 2],
        "classifier": {
            "fc_out_features": num_classes,
        },
        "parameter_count": {
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
    }


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)
    prepare_output_dir(OUTPUT_DIR, clear_old=CLEAR_OLD_OUTPUT)

    run_config = {
        "data_root": DATA_ROOT,
        "train_dir": TRAIN_DIR,
        "val_dir": VAL_DIR,
        "output_dir": OUTPUT_DIR,
        "clear_old_output": CLEAR_OLD_OUTPUT,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "device": DEVICE,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "criterion": "CrossEntropyLoss",
    }
    save_json(run_config, os.path.join(OUTPUT_DIR, "run_config.json"))

    print("DEVICE:", DEVICE)
    print("DATA_ROOT:", DATA_ROOT)

    train_dataset, val_dataset, train_loader, val_loader, class_names, num_classes, dataset_info = build_dataloaders()

    save_json(dataset_info, os.path.join(OUTPUT_DIR, "dataset_info.json"))

    print(f"Train size : {len(train_dataset)}")
    print(f"Val size   : {len(val_dataset)}")
    print(f"Class names: {class_names}")
    print(f"Class map  : {train_dataset.class_to_idx}")

    small_cnn = SmallCNN(num_classes=num_classes)
    resnet18_scratch = build_resnet18_scratch(num_classes=num_classes)

    small_cnn_config = get_smallcnn_model_config(small_cnn, num_classes)
    resnet18_config = get_resnet18_model_config(resnet18_scratch, num_classes)

    architecture_rows = build_architecture_rows(
        small_cnn,
        resnet18_scratch,
        image_size=IMAGE_SIZE,
        num_classes=num_classes,
    )
    save_json(architecture_rows, os.path.join(OUTPUT_DIR, "architecture_table.json"))
    save_csv(architecture_rows, os.path.join(OUTPUT_DIR, "architecture_table.csv"))
    save_architecture_latex(architecture_rows, os.path.join(OUTPUT_DIR, "architecture_table.tex"))

    cnn_summary = run_training(
        model_name="small_cnn_16_32_64",
        model=small_cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        class_names=class_names,
        model_config=small_cnn_config,
    )

    resnet_summary = run_training(
        model_name="resnet18_scratch",
        model=resnet18_scratch,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        class_names=class_names,
        model_config=resnet18_config,
    )

    final_compare = {
        "small_cnn_16_32_64": cnn_summary,
        "resnet18_scratch": resnet_summary,
    }

    save_json(final_compare, os.path.join(OUTPUT_DIR, "final_compare.json"))
    plot_final_compare(
        final_compare,
        os.path.join(OUTPUT_DIR, "final_compare.png"),
    )

    print("\n========== FINAL COMPARE ==========")
    for name, info in final_compare.items():
        print(
            f"{name}: "
            f"val_acc={info['best_val_acc']:.4f}, "
            f"val_macro_f1={info['best_val_macro_f1']:.4f}"
        )

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()