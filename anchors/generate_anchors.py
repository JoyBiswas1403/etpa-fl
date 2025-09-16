import os
import torch
import argparse
from torchvision import datasets, transforms

def generate_mnist_anchors(num_samples=1000, data_dir="./data"):
    """
    Generate anchor data by sampling from MNIST train set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    mnist = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    indices = torch.randperm(len(mnist))[:num_samples]
    x = torch.stack([mnist[i][0] for i in indices])
    y = torch.tensor([mnist[i][1] for i in indices])
    return x, y

def save_anchors(x, y, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({'anchors': x, 'labels': y}, output_path)
    print(f" Anchors saved at: {output_path} (keys: 'anchors', 'labels')")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MNIST-based anchors")
    parser.add_argument("--output", type=str, default="anchors/test_anchors.pt",
                        help="Path to save generated anchors")
    parser.add_argument("--num_samples", type=int, default=1200,
                        help="Number of anchor samples")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to download/load MNIST data")

    args = parser.parse_args()

    x, y = generate_mnist_anchors(num_samples=args.num_samples, data_dir=args.data_dir)
    x = torch.clamp(x, 0, 1)
    # Optional: smooth anchors with Gaussian blur
    # from torchvision.transforms import GaussianBlur
    # blur = GaussianBlur(kernel_size=3, sigma=1.0)
    # x = blur(x)

    save_anchors(x, y, args.output)