import torch
import flwr as fl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.simple_cnn import SimpleCNN
from utils.train_utils import train_dp
from client.dp_vae import DPVAE, train_dpvai, sample_anchors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset (dummy: MNIST for now, FEMNIST later)
# ----------------------------
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    trainset, valset = random_split(dataset, [train_len, val_len])
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(valset, batch_size=32)
    return trainloader, valloader

trainloader, valloader = load_data()

# ----------------------------
# Training & Eval
# ----------------------------
def test(model, valloader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    accuracy = correct / len(valloader.dataset)
    return loss, accuracy

# ----------------------------
# Flower client
# ----------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # ----------------------------
        # Local DP-SGD training
        # ----------------------------
        train_dp(self.model, trainloader, DEVICE, epochs=1, noise_multiplier=1.0)

        # ----------------------------
        # Train DP-VAE (anchors)
        # ----------------------------
        vae = DPVAE().to(DEVICE)
        vae = train_dpvai(vae, trainloader, DEVICE, epochs=1)
        anchors = sample_anchors(vae, num_samples=10, device=DEVICE)
        print("Generated anchors:", anchors.shape)

        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, valloader)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    model = SimpleCNN().to(DEVICE)
    client = FlowerClient(model)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
