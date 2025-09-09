# ETPA/client/dp_train.py

import torch
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm

# --- Configuration Constants ---
# These would typically be loaded from a config file
TARGET_EPSILON = 5.0
TARGET_DELTA = 1e-5 # Standard delta for image datasets
MAX_GRAD_NORM = 1.0 # Max L2 norm of per-sample gradients
EPOCHS = 3
BATCH_SIZE = 32

def train_dp(model, train_loader, optimizer, device):
    """
    Trains a model with Differential Privacy using Opacus.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
    """
    model.to(device)
    model.train()

    # 1. --- Attach Opacus PrivacyEngine ---
    # This is the core step for making training differentially private.
    # It hooks into the optimizer and model to perform DP-SGD.
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=TARGET_EPSILON,
        target_delta=TARGET_DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM} to achieve ε={TARGET_EPSILON}")

    # --- Standard Training Loop ---
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # Using BatchMemoryManager for efficient training with virtual batches if physical batch size is too large
        with BatchMemoryManager(
            data_loader=train_loader, 
            max_physical_batch_size=BATCH_SIZE // 2, 
            optimizer=optimizer
        ) as memory_safe_loader:
            for images, labels in memory_safe_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

        # 2. --- Get and Log Epsilon ---
        # The privacy engine keeps track of the privacy budget spent.
        epsilon = privacy_engine.get_epsilon(TARGET_DELTA)
        print(
            f"Epoch: {epoch+1} | "
            f"Loss: {loss.item():.4f} | "
            f"ε: {epsilon:.2f} (δ: {TARGET_DELTA})"
        )

    # It's good practice to detach the privacy engine when done
    privacy_engine.detach()
    return model

# Example of how this function might be called from client/main.py
if __name__ == '__main__':
    # This is a placeholder for demonstration.
    # In the actual project, Member A (Federated Core) would call train_dp from main.py.
    
    # 1. Setup dummy model, data, and optimizer
    # (Assuming Member B's data_loader and Member A's model are available)
    dummy_model = torch.nn.Sequential(torch.nn.Linear(10, 2))
    dummy_data = [(torch.randn(10), torch.randint(0, 2, (1,)).item()) for _ in range(200)]
    dummy_loader = DataLoader(dummy_data, batch_size=BATCH_SIZE)
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.05)
    
    print("--- Running DP Training Example ---")
    trained_model = train_dp(dummy_model, dummy_loader, dummy_optimizer, device='cpu')
    print("--- DP Training Complete ---")