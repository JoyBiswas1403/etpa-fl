def run_server():
    print("Server is running and waiting for clients...")
    # Step 1: Test multiple noise multipliers
    noise_multipliers = [0.5, 1.0, 2.0]
    tau = 0.1  # Threshold for event trigger (can be made configurable)
    print(f"Event trigger threshold (tau): {tau}\n")
    for noise in noise_multipliers:
        print(f"=== Testing with noise_multiplier={noise} ===")
        # Simulate running client training (in real use, call client script with noise param)
        # Here, we simulate divergence values for each noise setting
        if noise == 0.5:
            divergence_values = [0.05, 0.07, 0.09]
        elif noise == 1.0:
            divergence_values = [0.08, 0.12, 0.10]
        else:
            divergence_values = [0.15, 0.18, 0.22]
        for i, div in enumerate(divergence_values):
            print(f"Received divergence from client {i}: {div:.4f}")
            if div > tau:
                print(f"[EVENT TRIGGERED] Client {i} divergence {div:.4f} exceeds threshold tau={tau}")
                # Step 2: Simulate anchor regeneration
                print(f"[ANCHOR REGENERATION] Regenerating anchors due to high divergence...")
                # In real use, call: python etpa-fl/anchors/generate_anchors.py --output anchors/test_anchors.pt
        print()
    print("Server event trigger and anchor regeneration test complete.")

    if __name__ == "__main__":
        run_server()
