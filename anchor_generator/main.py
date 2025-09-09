# ETPA/anchor_generator/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import numpy as np
from tqdm import tqdm

# --- VAE and DP Configuration ---
IMAGE_SIZE = 28 * 28
H_DIM = 256
Z_DIM = 32 # Latent space dimension
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 64

# --- DP Parameters ---
DP_TARGET_EPSILON = 8.0
DP_TARGET_DELTA = 1e-5
DP_MAX_GRAD_NORM = 1.2

class VAE(nn.Module):
    """A simple VAE for image data like FEMNIST."""
    def __init__(self, image_size=IMAGE_SIZE, h_dim=H_DIM, z_dim=Z_DIM):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim) # mu
        self.fc3 = nn.Linear(h_dim, z_dim) # log_var

        # Decoder
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

def vae_loss_function(x_reconst, x, mu, log_var):
    """Calculates VAE loss = Reconstruction Loss + KL Divergence."""
    reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconst_loss + kl_div

def train_dp_vae(vae_model, data_loader, device):
    """Trains the VAE with differential privacy."""
    vae_model.to(device)
    vae_model.train()
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=LR)

    # --- Attach Opacus PrivacyEngine ---
    privacy_engine = PrivacyEngine()
    vae_model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=vae_model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=EPOCHS,
        target_epsilon=DP_TARGET_EPSILON,
        target_delta=DP_TARGET_DELTA,
        max_grad_norm=DP_MAX_GRAD_NORM,
    )

    print(f"Training DP-VAE to achieve ε={DP_TARGET_EPSILON}...")

    for epoch in range(EPOCHS):
        for i, (images, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}")):
            images = images.view(-1, IMAGE_SIZE).to(device)
            
            # Forward pass
            x_reconst, mu, log_var = vae_model(images)
            loss = vae_loss_function(x_reconst, images, mu, log_var)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon = privacy_engine.get_epsilon(DP_TARGET_DELTA)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item()/len(images):.4f}, ε: {epsilon:.2f}")

    privacy_engine.detach()
    return vae_model

def generate_anchors(vae_model, num_samples, device):
    """Generates synthetic anchor samples from the trained VAE."""
    vae_model.eval()
    with torch.no_grad():
        # Sample from the latent space (standard normal distribution)
        z = torch.randn(num_samples, Z_DIM).to(device)
        # Decode the latent vectors to generate samples
        generated_samples = vae_model.decode(z).cpu()
    return generated_samples.view(-1, 1, 28, 28) # Reshape to image format

def save_anchors(anchors, path="dp_anchors.npz"):
    """Saves the generated anchors to a file."""
    # Convert to numpy and save as a compressed .npz file
    np.savez_compressed(path, data=anchors.numpy())
    print(f"Saved {len(anchors)} anchors to {path}")

if __name__ == '__main__':
    # --- This is a placeholder for a real dataset loader (from Member B) ---
    # Generating dummy data for demonstration (e.g., FEMNIST would be 28x28)
    dummy_femnist_data = [(torch.rand(1, 28, 28), 0) for _ in range(1024)]
    dummy_loader = DataLoader(dummy_femnist_data, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize and Train the DP-VAE
    vae = VAE().to(device)
    trained_vae = train_dp_vae(vae, dummy_loader, device)

    # 2. Generate and Save Anchors
    num_anchors_to_generate = 500
    dp_anchors = generate_anchors(trained_vae, num_anchors_to_generate, device)
    save_anchors(dp_anchors)

    # 3. (Optional) Quick visual check of one anchor
    try:
        import matplotlib.pyplot as plt
        print("Displaying one generated anchor sample.")
        plt.imshow(dp_anchors[0].squeeze(), cmap='gray')
        plt.title("Generated DP Anchor")
        plt.show()
    except ImportError:
        print("Matplotlib not found. Skipping visualization.")