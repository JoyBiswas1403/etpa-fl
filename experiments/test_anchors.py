# experiments/test_anchors.py
"""
Quick anchor validation script for ETPA project.

Usage examples (PowerShell):
  python experiments/test_anchors.py --anchors anchors/test_anchors.pt --dataset FEMNIST --visualize --num-samples 36
  python experiments/test_anchors.py --anchors anchors/test_anchors.npz --dataset FEMNIST --classifier-epochs 5

What it does:
 - loads anchors (torch .pt or numpy .npz)
 - prints basic stats (shape, dtype, mean/std, class distribution if labels present)
 - visualizes a sample grid of images
 - if anchors have labels: trains a tiny CNN on anchors and evaluates on a small real test loader
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# Attempt to import your utils (adjust import if your utils module differs)
try:
    from utils.data_utils import get_dataloader
    from utils.metrics import compute_accuracy
except Exception:
    # fallback simple get_dataloader if utils not present
    def get_dataloader(dataset_name="FEMNIST", batch_size=128, train=False):
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        if dataset_name.upper().startswith("FEMNIST") or dataset_name.upper().startswith("MNIST"):
            ds = datasets.FashionMNIST("./data", train=train, download=True, transform=transform) if False else datasets.MNIST("./data", train=train, download=True, transform=transform)
        else:
            ds = datasets.MNIST("./data", train=train, download=True, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def compute_accuracy(model, dataloader, device="cpu"):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return 100.0 * correct / total

# small CNN classifier (tiny to be fast)
class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128) if in_channels==1 else nn.Linear(32*8*8,128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def load_anchors(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Anchor file not found: {path}")
    if path.suffix == ".pt" or path.suffix == ".pth":
        obj = torch.load(path, map_location="cpu")
        # Try common keys
        if isinstance(obj, dict):
            for key in ("anchors", "data", "images", "x"):
                if key in obj:
                    data = obj[key]
                    break
            else:
                # last resort: first tensor-like item
                vals = [v for v in obj.values() if isinstance(v, (torch.Tensor, np.ndarray))]
                data = vals[0] if vals else None
            labels = obj.get("labels", obj.get("y", None))
        else:
            data = obj
            labels = None
        # Debug: print type and shape of data before tensor conversion
        print("[DEBUG] type(data):", type(data))
        if hasattr(data, 'shape'):
            print("[DEBUG] data.shape:", getattr(data, 'shape', None))
        else:
            print("[DEBUG] data has no 'shape' attribute")
        try:
            if isinstance(data, torch.Tensor):
                x = data.cpu()
            else:
                x = torch.tensor(data)
        except Exception as e:
            print("[ERROR] Failed to convert data to torch tensor:", e)
            print("[ERROR] Data content (truncated):", str(data)[:500])
            raise ValueError("Anchor file format is not compatible. Please check the anchor generation script and ensure it saves a tensor or numpy array under a common key like 'anchors', 'data', 'images', or 'x'.") from e
        if labels is not None:
            labels = torch.tensor(labels)
        return x, labels
    elif path.suffix in (".npz", ".npz"):
        npz = np.load(path, allow_pickle=True)
        # common names
        if "images" in npz:
            x = torch.tensor(npz["images"])
        elif "data" in npz:
            x = torch.tensor(npz["data"])
        elif "arr_0" in npz:
            x = torch.tensor(npz["arr_0"])
        else:
            # take first array
            k = list(npz.keys())[0]
            x = torch.tensor(npz[k])
        labels = None
        if "labels" in npz:
            labels = torch.tensor(npz["labels"])
        return x, labels
    else:
        raise ValueError("Unsupported anchor file type. Use .pt, .pth, or .npz")

def show_image_grid(images, ncols=6, title=None):
    images = images.detach().cpu().numpy()
    n = images.shape[0]
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(ncols*1.5, nrows*1.5))
    for i in range(n):
        ax = plt.subplot(nrows, ncols, i+1)
        img = images[i]
        # If flat vector, reshape to 28x28 for MNIST-like images
        if img.ndim == 1 and img.shape[0] == 28*28:
            img = img.reshape(28, 28)
        # handle (C,H,W) or (H,W)
        if img.ndim == 3:
            img = np.transpose(img, (1,2,0))
            if img.shape[-1] == 1:
                img = img[..., 0]
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.axis('off')
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def simple_train_eval(anchors_x, anchors_y, test_loader, epochs=3, device="cpu"):
    # anchors_x: tensor shape (N,C,H,W), (N,H,W), or (N,784)
    if anchors_x.ndim == 2 and anchors_x.shape[1] == 28*28:
        anchors_x = anchors_x.view(-1, 1, 28, 28)
    elif anchors_x.ndim == 3:
        anchors_x = anchors_x.unsqueeze(1)  # (N,1,H,W)
    in_ch = anchors_x.shape[1]
    num_classes = int(anchors_y.max().item()) + 1
    model = SmallCNN(in_channels=in_ch, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(anchors_x, anchors_y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(loader.dataset)
        print(f"[anchor-train] epoch {ep+1}/{epochs} loss={avg:.4f}")

    acc = compute_accuracy(model, test_loader, device=device)
    print(f"Classifier trained on anchors -> test accuracy: {acc:.2f}%")
    return model, acc

def main(args):
    device = "cuda" if torch.cuda.is_available() and args.device!="cpu" else "cpu"
    x, y = load_anchors(args.anchors)
    print(f"Loaded anchors: x.shape={tuple(x.shape)}, labels={'present' if y is not None else 'absent'}")

    # normalize to [0,1] if needed
    if x.dtype in (torch.uint8,):
        x = x.float() / 255.0
    else:
        x = x.float()

    # If shape is (N,H,W) -> convert to (N,1,H,W)
    if x.ndim == 3:
        x = x.unsqueeze(1)

    # Basic stats
    print("Stats: min %.4f max %.4f mean %.4f std %.4f" % (x.min().item(), x.max().item(), x.mean().item(), x.std().item()))
    if y is not None:
        unique, counts = torch.unique(y, return_counts=True)
        print("Label distribution:")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"  {u}: {c}")

    # Visualize samples
    ns = min(args.num_samples, x.shape[0])
    show_image_grid(x[:ns], ncols=6, title="Sample anchors")

    # If labels available and user asked for classifier evaluation:
    if y is not None and args.classifier_epochs > 0:
        print("Loading a small test loader from dataset:", args.dataset)
        test_loader = get_dataloader(dataset_name=args.dataset, batch_size=args.test_batch, train=False)
        # ensure shapes: convert test images to match channels (if test images are (N,1,H,W) OK)
        model, acc = simple_train_eval(x, y, test_loader, epochs=args.classifier_epochs, device=device)
    else:
        if y is None:
            print("No labels present in anchors; skipping classifier test.")
        else:
            print("Classifier epochs set to 0; skipping classifier evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=str, required=True, help="Path to anchors file (.pt or .npz)")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset name for test loader")
    parser.add_argument("--visualize", action="store_true", help="Show image grid")
    parser.add_argument("--num-samples", type=int, default=36, help="Number of samples to visualize")
    parser.add_argument("--classifier-epochs", type=int, default=3, help="Epochs to train small classifier on anchors (if labels present)")
    parser.add_argument("--test-batch", type=int, default=256, help="Batch size for test loader")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'")
    args = parser.parse_args()
    main(args)
