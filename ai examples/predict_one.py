import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# =========================
# CLASS NAMES (SABİT)
# =========================
CLASS_NAMES = [
    "badlands",
    "birch_forest",
    "cherry_grove",
    "dark_forest",
    "desert",
    "flower_forest",
    "forest",
    "frozen_peaks",
    "ice_spikes",
    "jungle",
    "mushroom_fields",
    "ocean",
    "pale_garden",
    "plains",
    "savanna",
    "snowy_plains",
    "stony_shore",
    "sunflower_plains",
    "taiga",
]


def build_model(num_classes: int):
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    return model


def get_transform(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


@torch.no_grad()
def predict_image(model, image_path, tfm, class_names, device, topk=5):
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(topk, len(class_names))
    top_probs, top_idx = torch.topk(probs, k=k)

    results = []
    for p, i in zip(top_probs.cpu().tolist(), top_idx.cpu().tolist()):
        results.append((class_names[i], p))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Tahmin edilecek resim yolu"
    )
    parser.add_argument(
        "--ckpt",
        default="best_biome_resnet18.pth",
        help="Model checkpoint yolu"
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tfm = get_transform(img_size=args.img_size)

    results = predict_image(
        model=model,
        image_path=args.image,
        tfm=tfm,
        class_names=CLASS_NAMES,
        device=device,
        topk=args.topk,
    )

    print(f"\nImage: {args.image}")
    print("Top predictions:")
    for rank, (name, prob) in enumerate(results, start=1):
        print(f"{rank:>2}. {name:<20}  {prob*100:.2f}%")

    print(f"\nPredicted biome (top-1): {results[0][0]}")


if __name__ == "__main__":
    main()
