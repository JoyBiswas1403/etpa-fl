import torch
from opacus import PrivacyEngine

# ----------------------------
# Local DP-SGD training (for FL client)
# ----------------------------
def train_dp(model, trainloader, device, epochs=1, lr=0.01, noise_multiplier=1.0, max_grad_norm=1.0):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    model.train()
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model
