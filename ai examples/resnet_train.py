import os
import copy
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt


# --------------------------
# Utils
# --------------------------
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """
    Val loss iyileşmezse durdurur.
    """
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        # True dönerse "stop"
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def plot_confusion_matrix(cm, class_names, normalize=True, title="Confusion Matrix", save_path=None):
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


# --------------------------
# Main train/eval
# --------------------------
def main(
    data_dir="data",
    img_size=224,
    batch_size=32,
    num_epochs=40,
    lr=3e-4,
    weight_decay=1e-4,
    val_size=0.15,
    test_size=0.15,
    patience=7,
    seed=42,
    num_workers=2,
    use_amp=True,
    unfreeze_last_block=True,
):
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Transforms (ImageNet stats)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # İstersen aç:
        # transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value='random'),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Aynı klasörü iki farklı transform ile kullanmak için iki dataset objesi:
    base_dataset = datasets.ImageFolder(root=data_dir)
    train_dataset_full = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    eval_dataset_full  = datasets.ImageFolder(root=data_dir, transform=eval_tfms)

    class_names = base_dataset.classes
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Num classes:", num_classes)

    # Stratified split: train/val/test
    targets = np.array([y for _, y in base_dataset.samples])
    indices = np.arange(len(targets))

    # önce test ayır
    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=targets
    )
    # train/val ayır
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_size / (1.0 - test_size),
        random_state=seed, stratify=targets[trainval_idx]
    )

    train_ds = Subset(train_dataset_full, train_idx)
    val_ds   = Subset(eval_dataset_full,  val_idx)
    test_ds  = Subset(eval_dataset_full,  test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Model: ResNet18 pretrained
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Optionally unfreeze last block for better accuracy
    if unfreeze_last_block:
        for p in model.layer4.parameters():
            p.requires_grad = True

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Scheduler: val loss düşmezse LR azalt
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    early_stopper = EarlyStopping(patience=patience, min_delta=1e-4)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_acc += accuracy_from_logits(outputs.detach(), labels)
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        train_acc = running_acc / max(n_batches, 1)

        # ---- Val ----
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item()
                val_acc_sum += accuracy_from_logits(outputs, labels)
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        val_acc = val_acc_sum / max(val_batches, 1)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"time={dt:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "best_biome_resnet18.pth")

        # Early stop check
        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
            break

    # Load best
    model.load_state_dict(best_model_wts)

    # ---- Test + Confusion Matrix ----
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    test_acc = (all_preds == all_true).mean()
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    print("\nClassification report:")
    print(classification_report(all_true, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_true, all_preds)
    # Ham CM
    np.save("cm_resnet18_raw.npy", cm)
    plot_confusion_matrix(
        cm, class_names,
        normalize=False,
        title="ResNet18 Confusion Matrix (Raw)",
        save_path="cm_resnet18_raw.png"
    )

    # Normalize CM
    plot_confusion_matrix(
        cm, class_names,
        normalize=True,
        title="ResNet18 Confusion Matrix (Normalized)",
        save_path="cm_resnet18_norm.png"
    )

    # ---- Training curves ----
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # data_dir'i kendi klasör adına göre düzenle (ör: "./data")
    main(
        data_dir="./data",
        batch_size=32,
        num_epochs=40,
        lr=3e-4,
        patience=7,
        use_amp=True,
        unfreeze_last_block=True,
    )
