import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names, normalize=True, title="Confusion Matrix", save_path=None):
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
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


def build_xy(subset, img_size=64):
    """
    subset: Subset(ImageFolder)
    Görüntüyü resize edip flatten -> X, label -> y
    """
    ds = subset.dataset  # ImageFolder (transform yok)
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # 0..1
    ])

    X, y = [], []
    for idx in subset.indices:
        path, label = ds.samples[idx]
        img = datasets.folder.default_loader(path)  # PIL
        t = tfm(img)  # 3 x H x W
        X.append(t.reshape(-1).numpy())
        y.append(label)

    return np.stack(X), np.array(y)


def main(
    data_dir="./data",
    img_size=64,
    test_size=0.15,
    val_size=0.15,   # DT için val şart değil ama aynı split mantığı için tutuyoruz
    seed=42,
    max_depth=25
):
    base_ds = datasets.ImageFolder(root=data_dir)  # transform yok
    class_names = base_ds.classes
    targets = np.array([y for _, y in base_ds.samples])
    indices = np.arange(len(targets))

    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=targets
    )
    train_idx, _val_idx = train_test_split(
        trainval_idx, test_size=val_size / (1.0 - test_size),
        random_state=seed, stratify=targets[trainval_idx]
    )

    train_subset = Subset(base_ds, train_idx)
    test_subset  = Subset(base_ds, test_idx)

    print("Classes:", class_names)
    print("Num classes:", len(class_names))
    print("Building flattened pixel features... (this can take a bit)")

    X_train, y_train = build_xy(train_subset, img_size=img_size)
    X_test,  y_test  = build_xy(test_subset,  img_size=img_size)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=seed
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = (y_pred == y_test).mean()
    print(f"\nDecision Tree Test Accuracy: {acc*100:.2f}%")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    np.save("cm_decision_tree_raw.npy", cm)

    plot_confusion_matrix(cm, class_names, normalize=False,
                          title="Decision Tree Confusion Matrix (Raw)",
                          save_path="cm_decision_tree_raw.png")
    plot_confusion_matrix(cm, class_names, normalize=True,
                          title="Decision Tree Confusion Matrix (Normalized)",
                          save_path="cm_decision_tree_norm.png")


if __name__ == "__main__":
    main()
