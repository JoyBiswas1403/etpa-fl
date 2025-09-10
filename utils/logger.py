import csv
import os

def log_results(filename, row, headers=None):
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists and headers:
            writer.writerow(headers)
        writer.writerow(row)

def print_progress(round_num, acc, loss):
    print(f"[Round {round_num}] Accuracy: {acc:.2f}%, Loss: {loss:.4f}")
