from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import argparse


# ----------------------------
# CNN Model for Better Accuracy
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # MNIST/FEMNIST shape
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ----------------------------
# Divergence Metric
# ----------------------------
def compute_divergence(prev_params, new_params):
    prev_vector = torch.cat([p.view(-1) for p in prev_params.values()])
    new_vector = torch.cat([p.view(-1) for p in new_params.values()])
    cosine_similarity = torch.nn.functional.cosine_similarity(prev_vector, new_vector, dim=0)
    cosine_div = 1.0 - cosine_similarity.item()
    l2_div = torch.norm(prev_vector - new_vector, p=2).item()
    return cosine_div, l2_div


# ----------------------------
# Dummy Private Data Loader
# Replace this with FEMNIST/Shakespeare loader later
# ----------------------------
def load_data():
    X_train = torch.randn(100, 1, 28, 28)
    y_train = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return train_loader


# ----------------------------
# Event Trigger Rule
# ----------------------------
def adjust_noise(noise_multiplier, cosine_div, threshold=0.3):
    if cosine_div > threshold:
        return noise_multiplier * 1.1  # more noise
    else:
        return max(noise_multiplier * 0.95, 0.5)  # reduce noise but keep >=0.5


# ----------------------------
# Client Training
# ----------------------------
def run_client(client_id: int):
    print(f"Client {client_id} is running...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 15
    alpha = 0.8
    noise_multiplier = 1.0
    delta = 1e-5

    # Private Data
    train_loader = load_data()

    # Anchors
    anchor_path = f"anchors/client_{client_id}.pt"
    try:
        anchor_obj = torch.load(anchor_path, map_location=device)
        if isinstance(anchor_obj, dict):
            anchor_x = anchor_obj.get("anchors")
            anchor_y = anchor_obj.get("labels")
        else:
            anchor_x, anchor_y = anchor_obj
        print(f"Client {client_id}: Loaded {len(anchor_x)} anchors.")
        anchor_dataset = TensorDataset(anchor_x, anchor_y)
        anchor_loader = DataLoader(anchor_dataset, batch_size=32, shuffle=True)
    except Exception as e:
        print(f"Client {client_id}: No anchors found. Error: {e}")
        anchor_loader = None

    # Attach Privacy Engine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=1.0,
    )

    previous_model_state = deepcopy(model.state_dict())

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        anchor_iter = iter(anchor_loader) if anchor_loader is not None else None

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            out_priv = model(data)
            loss_priv = criterion(out_priv, target)

            # Anchors
            if anchor_iter is not None:
                try:
                    anchor_data, anchor_target = next(anchor_iter)
                except StopIteration:
                    anchor_iter = iter(anchor_loader)
                    anchor_data, anchor_target = next(anchor_iter)
                anchor_data, anchor_target = anchor_data.to(device), anchor_target.to(device)
                out_anchor = model(anchor_data)
                loss_anchor = criterion(out_anchor, anchor_target)
            else:
                loss_anchor = torch.tensor(0.0, device=device)

            # Weighted loss
            loss = alpha * loss_priv + (1 - alpha) * loss_anchor
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Metrics
        avg_loss = epoch_loss / len(train_loader)
        epsilon = privacy_engine.get_epsilon(delta)
        current_model_state = model.state_dict()
        cosine_div, l2_div = compute_divergence(previous_model_state, current_model_state)
        previous_model_state = deepcopy(current_model_state)

        # Adaptive noise
        noise_multiplier = adjust_noise(noise_multiplier, cosine_div)
        print(
            f"Client {client_id} | Epoch {epoch+1}/{num_epochs} "
            f"| Loss: {avg_loss:.4f} | ε: {epsilon:.2f} "
            f"| CosDiv: {cosine_div:.4f} | L2Div: {l2_div:.4f} "
            f"| Noise: {noise_multiplier:.2f}"
        )

    print(f"✅ Client {client_id} finished training with final ε = {epsilon:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=0, help="Client ID (for multi-client runs)")
    args = parser.parse_args()
    run_client(args.client_id)
