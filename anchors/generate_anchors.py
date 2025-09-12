import os
import torch
import argparse

def generate_dummy_anchors(num_samples=1000, input_dim=28*28, num_classes=10):
    """
    Generate dummy anchor data for testing.
    - Features: random tensors
    - Labels: random integers between 0 and num_classes-1
    """
    x = torch.randn(num_samples, input_dim)   # random features
    y = torch.randint(0, num_classes, (num_samples,))  # random labels
    return x, y


def save_anchors(x, y, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({'anchors': x, 'labels': y}, output_path)
    print(f" Anchors saved at: {output_path} (keys: 'anchors', 'labels')")


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
