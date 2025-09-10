from client.train_client import run_client
from server.server import run_server

if __name__ == "__main__":
    print("🚀 Starting Event-Triggered Privacy Anchors (ETPA) Simulation")

    # Start server first
    run_server()

    # Start one or more clients (later you’ll scale this)
    run_client(client_id=1)
    run_client(client_id=2)
