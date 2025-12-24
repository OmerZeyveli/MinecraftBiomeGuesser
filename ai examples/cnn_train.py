import copy
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0

    def step(self, val_loss: float) -> bool:
        if self.best is None:
            self.best = val_loss
            return False
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


def plot_confusion_matrix(cm, class_names, normalize=True, title="Confusion Matrix", save_path=None):
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


class ScratchCNN(nn.Module):
    """
    From-scratch CNN (pretrained yok).
    Input: 3 x img_size x img_size
    """
    def __init__(self, num_classes: int, img_size: int = 128):
        super().__init__()
        self.img_size = img_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /16
        )

        fm = img_size // 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * fm * fm, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def main(
    data_dir="./data",
    img_size=128,
    batch_size=32,
    num_epochs=60,
    lr=3e-4,
    weight_decay=1e-4,
    val_size=0.15,
    test_size=0.15,
    patience=10,
    seed=42,
    num_workers=2,
    use_amp=True,
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    base_dataset = datasets.ImageFolder(root=data_dir)
    train_dataset_full = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    eval_dataset_full  = datasets.ImageFolder(root=data_dir, transform=eval_tfms)

    class_names = base_dataset.classes
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Num classes:", num_classes)

    targets = np.array([y for _, y in base_dataset.samples])
    indices = np.arange(len(targets))

    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=targets
    )
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

    model = ScratchCNN(num_classes=num_classes, img_size=img_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    early = EarlyStopping(patience=patience, min_delta=1e-4)

    best_wts = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        model.train()
        tr_loss, tr_acc, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item()
            tr_acc += accuracy_from_logits(logits.detach(), y)
            n += 1

        tr_loss /= max(n, 1)
        tr_acc  /= max(n, 1)

        model.eval()
        va_loss, va_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += loss.item()
                va_acc  += accuracy_from_logits(logits, y)
                m += 1

        va_loss /= max(m, 1)
        va_acc  /= max(m, 1)

        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | "
              f"lr={lr_now:.2e} | time={time.time()-t0:.1f}s")

        if va_loss < best_val:
            best_val = va_loss
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, "best_scratch_cnn.pth")

        if early.step(va_loss):
            print(f"Early stopping triggered at epoch {epoch}. Best val_loss={best_val:.4f}")
            break

    model.load_state_dict(best_wts)

    # Test + CM
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)

    test_acc = (all_preds == all_true).mean()
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print("\nClassification report:")
    print(classification_report(all_true, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_true, all_preds)
    np.save("cm_scratch_cnn_raw.npy", cm)
    plot_confusion_matrix(cm, class_names, normalize=False,
                          title="Scratch CNN Confusion Matrix (Raw)",
                          save_path="cm_scratch_cnn_raw.png")
    plot_confusion_matrix(cm, class_names, normalize=True,
                          title="Scratch CNN Confusion Matrix (Normalized)",
                          save_path="cm_scratch_cnn_norm.png")


if __name__ == "__main__":
    main()
