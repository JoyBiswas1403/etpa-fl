from client.train_client import run_client
from server.server import run_server

if __name__ == "__main__":
    print("Starting a test run with a single client...")

    # --- We are commenting out the server for this test ---
    # run_server()

    # --- We will run ONLY ONE client to see its full output ---
    run_client(client_id=1)

    # --- We are commenting out the second client for this test ---
    # run_client(client_id=2)

    print("Test run finished.")