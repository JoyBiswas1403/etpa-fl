import torch
import torch.optim as optim
import torch.nn.functional as F

def train_one_client(model, train_loader, epochs=1, lr=0.01, device="cpu"):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()  # return updated weights
