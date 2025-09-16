import os
import torch
import argparse
<<<<<<< HEAD
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

=======

def generate_dummy_anchors(num_samples=1000, input_dim=28*28, num_classes=10):
    """
    Generate dummy anchor data for testing.
    - Features: random tensors
    - Labels: random integers between 0 and num_classes-1
    """
    x = torch.randn(num_samples, input_dim)   # random features
    y = torch.randint(0, num_classes, (num_samples,))  # random labels
    return x, y


>>>>>>> 53322fef5a934a379f580fc7bd3ed60a5a8d40f6
def save_anchors(x, y, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({'anchors': x, 'labels': y}, output_path)
    print(f" Anchors saved at: {output_path} (keys: 'anchors', 'labels')")

<<<<<<< HEAD
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
=======

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic anchors")
    parser.add_argument("--output", type=str, default="anchors/test_anchors.pt",
                        help="Path to save generated anchors")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of synthetic samples")
    parser.add_argument("--input_dim", type=int, default=28*28,
                        help="Input dimension (flattened image size)")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes")

    args = parser.parse_args()

    # Generate dummy anchors
    x, y = generate_dummy_anchors(args.num_samples, args.input_dim, args.num_classes)

    # Save them
    save_anchors(x, y, args.output)
>>>>>>> 53322fef5a934a379f580fc7bd3ed60a5a8d40f6
