import numpy as np
import torch
from torchvision import datasets, transforms

def get_dataloader(dataset_name="MNIST", batch_size=32, train=True):
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == "MNIST":
        dataset = datasets.MNIST("./data", train=train, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10("./data", train=train, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported yet!")

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def partition_data_noniid(dataset, num_clients, alpha=0.5):
    """
    Dirichlet distribution for non-IID partitioning.
    alpha < 1 => more skewed distribution
    """
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    idxs = np.arange(len(dataset))
    client_data = [[] for _ in range(num_clients)]

    proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients), size=num_classes)

    for c in range(num_classes):
        class_idx = idxs[labels == c]
        np.random.shuffle(class_idx)
        split = (proportions[c] / proportions[c].sum()).cumsum()[:-1]
        split_idx = (split * len(class_idx)).astype(int)
        client_chunks = np.split(class_idx, split_idx)
        for i in range(num_clients):
            client_data[i].extend(client_chunks[i])

    return client_data
