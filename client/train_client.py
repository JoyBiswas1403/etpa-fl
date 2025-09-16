from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import argparse



# Larger model for better learning

# Simpler model for better convergence
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compute_divergence(prev_params, new_params):
    """Computes cosine and L2 divergence between two sets of model parameters."""
    prev_vector = torch.cat([p.view(-1) for p in prev_params.values()])
    new_vector = torch.cat([p.view(-1) for p in new_params.values()])
    # Cosine divergence
    cosine_similarity = torch.nn.functional.cosine_similarity(prev_vector, new_vector, dim=0)
    cosine_div = 1.0 - cosine_similarity.item()
    # L2 divergence
    l2_div = torch.norm(prev_vector - new_vector, p=2).item()
    return cosine_div, l2_div


from torchvision import datasets, transforms

def load_data(data_dir="./data", num_samples=1000, batch_size=64):
    """Load MNIST data for the client."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    mnist = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    indices = torch.randperm(len(mnist))[:num_samples]
    X = torch.stack([mnist[i][0] for i in indices])
    y = torch.tensor([mnist[i][1] for i in indices])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


def run_client(client_id: int):
    """
    This function trains a single client with differential privacy.
    """
    print(f"Client {client_id} is running...")
    print(f"DEBUG: Client {client_id} is starting the full training process now...")


    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Even lower learning rate
    criterion = nn.CrossEntropyLoss()
    num_epochs = 80  # Even more epochs
    alpha = 0.95  # Even higher weight on private data
    batch_size = 128

    # Load private data
    train_loader = load_data(batch_size=batch_size)

    # Load anchor data
    anchor_path = 'anchors/test_anchors.pt'
    try:
        anchor_obj = torch.load(anchor_path, map_location='cpu')
        if isinstance(anchor_obj, dict):
            anchor_x = anchor_obj.get('anchors')
            anchor_y = anchor_obj.get('labels')
        else:
            anchor_x, anchor_y = anchor_obj
        print(f"Client {client_id}: Loaded anchor data from {anchor_path}")
        anchor_dataset = TensorDataset(anchor_x, anchor_y)
        anchor_loader = DataLoader(anchor_dataset, batch_size=16, shuffle=True)
    except Exception as e:
        print(f"Client {client_id}: Could not load anchors. Only private data will be used. Error: {e}")
        anchor_loader = None

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=0.5,  # Even lower noise for better utility
        max_grad_norm=1.0,
    )
    print(f"Client {client_id}: Attached PrivacyEngine.")

    previous_model_state = deepcopy(model.state_dict())
    model.train()
    for epoch in range(num_epochs):
        anchor_iter = iter(anchor_loader) if anchor_loader is not None else None
        for data, target in train_loader:
            optimizer.zero_grad()
            # Private data loss
            output_private = model(data)
            loss_private = criterion(output_private, target)

            # Anchor data loss (if available)
            if anchor_iter is not None:
                try:
                    anchor_data, anchor_target = next(anchor_iter)
                except StopIteration:
                    anchor_iter = iter(anchor_loader)
                    anchor_data, anchor_target = next(anchor_iter)
                output_anchor = model(anchor_data)
                loss_anchor = criterion(output_anchor, anchor_target)
            else:
                loss_anchor = torch.tensor(0.0)
            # Weighted loss
            loss = alpha * loss_private + (1 - alpha) * loss_anchor
            loss.backward()
            optimizer.step()

        delta = 1e-5
        epsilon = privacy_engine.get_epsilon(delta)
        current_model_state = model.state_dict()
        cosine_div, l2_div = compute_divergence(previous_model_state, current_model_state)
        print(
            f"Client {client_id} | Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {loss.item():.4f} | Epsilon: {epsilon:.2f} | "
            f"CosineDiv: {cosine_div:.4f} | L2Div: {l2_div:.4f}"
        )
        previous_model_state = deepcopy(current_model_state)
    print(f"Client {client_id} finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train client with DP and anchors")
    parser.add_argument('--client_id', type=int, default=0, help='Client ID (for multi-client runs)')
    args = parser.parse_args()
    run_client(args.client_id)