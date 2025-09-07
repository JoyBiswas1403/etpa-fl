# utils/data_loader.py

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def get_dataset(name, data_dir):
    transform = transforms.Compose([transforms.ToTensor()])

    if name.lower() == "mnist":
        train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    elif name.lower() == "emnist":
        train = datasets.EMNIST(root=data_dir, split="balanced", train=True, download=True, transform=transform)
        test = datasets.EMNIST(root=data_dir, split="balanced", train=False, download=True, transform=transform)

    elif name.lower() == "cifar10":
        transform_cifar = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_cifar)
        test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_cifar)
    else:
        raise NotImplementedError(f"Dataset {name} not supported yet!")

    return train, test


def dirichlet_split_noniid(train_dataset, n_clients, alpha, seed):
    np.random.seed(seed)
    n_classes = len(train_dataset.classes)
    targets = np.array(train_dataset.targets)
    client_indices = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        split_idx = np.split(idx_c, proportions)
        for i, idx in enumerate(split_idx):
            client_indices[i].extend(idx)

    for i in range(n_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def get_federated_dataloaders(dataset, data_dir, n_clients, alpha, batch_size, seed, partition_json=None):
    train_dataset, test_dataset = get_dataset(dataset, data_dir)

    if partition_json and os.path.exists(partition_json):
        with open(partition_json, "r") as f:
            client_indices = json.load(f)
        client_indices = {int(k): v for k, v in client_indices.items()}
    else:
        indices = dirichlet_split_noniid(train_dataset, n_clients, alpha, seed)
        client_indices = {i: idx for i, idx in enumerate(indices)}
        if partition_json:
            os.makedirs(os.path.dirname(partition_json), exist_ok=True)
            with open(partition_json, "w") as f:
                json.dump(client_indices, f)

    train_loaders = {}
    for cid, indices in client_indices.items():
        subset = Subset(train_dataset, indices)
        train_loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    meta = {"client_to_indices": client_indices, "stats": {cid: {"n_samples": len(idx)} for cid, idx in client_indices.items()}}
    return train_loaders, test_loader, meta


# CLI utility
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="emnist")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--n_clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition_json", type=str, default=None)
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()

    train_loaders, test_loader, meta = get_federated_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        n_clients=args.n_clients,
        alpha=args.alpha,
        batch_size=args.batch_size,
        seed=args.seed,
        partition_json=args.partition_json
    )

    print(f"Loaded {args.dataset} with {args.n_clients} clients (alpha={args.alpha})")
    if args.preview:
        for cid in range(min(args.preview, args.n_clients)):
            print(f"Client {cid}: {meta['stats'][cid]['n_samples']} samples")
