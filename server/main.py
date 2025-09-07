from utils.data_loader import get_federated_dataloaders
from utils.model import SimpleCNN
from client.trainer import train_one_client
import torch
import copy

def federated_averaging(global_model, client_loaders, rounds=1, device="cpu"):
    for r in range(rounds):
        print(f"\n--- Round {r+1} ---")
        client_weights = []
        for cid, loader in client_loaders.items():
            local_model = copy.deepcopy(global_model)
            new_weights = train_one_client(local_model, loader, epochs=1, device=device)
            client_weights.append(new_weights)

        # Average weights
        new_state_dict = copy.deepcopy(client_weights[0])
        for k in new_state_dict.keys():
            for i in range(1, len(client_weights)):
                new_state_dict[k] += client_weights[i][k]
            new_state_dict[k] = new_state_dict[k] / len(client_weights)

        global_model.load_state_dict(new_state_dict)
        print("✅ Global model updated")

    return global_model

def main():
    dataset = "mnist"
    data_dir = "./data"
    n_clients = 5
    alpha = 0.5
    batch_size = 32
    seed = 42

    train_loaders, test_loader, meta = get_federated_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        n_clients=n_clients,
        alpha=alpha,
        batch_size=batch_size,
        seed=seed
    )

    global_model = SimpleCNN()
    global_model = federated_averaging(global_model, train_loaders, rounds=2)

if __name__ == "__main__":
    main()

