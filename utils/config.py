import argparse

def get_config():
    parser = argparse.ArgumentParser(description="ETPA Federated Learning")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dp_epsilon", type=float, default=1.0)
    parser.add_argument("--dp_delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()
