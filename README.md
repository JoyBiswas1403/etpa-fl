# Event-Triggered Privacy Anchors (ETPA) for Federated Learning

## 📌 Project Overview
This project implements **Federated Learning with Differential Privacy (DP)** using two key ideas:
- **Privacy Anchors** → Synthetic samples generated via DP-VAE / DP-GAN  
- **Event-Triggered Adaptation** → Dynamically adjusting DP noise and regenerating anchors when client models diverge  

The aim is to achieve the right balance between **accuracy, privacy, and efficiency** in non-IID federated learning setups.

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/JoyBiswas1403/etpa-fl.git
cd etpa-fl

--- 

### 2. Setup Virtual Environment
```bash
python -m venv .venv

- Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1

- Linus/Mac:
```bash
source .venv/bin/activate

---

### 3. Install Dependencies 
```bash
pip install --upgrade pip
pip install -r requirements.txt

---

📂 Repository Structure
```bash

etpa-fl/
│── client/            # Client-side training logic
│── server/            # Server aggregation and coordination
│── utils/             # Data partitioning, evaluation, helper functions
│── anchors/           # Privacy Anchor generation (DP-VAE, DP-GAN)
│── requirements.txt   # Python dependencies
│── README.md          # Project documentation
│── main.py            # Entry point

---

##🚀Usage

### Run Full Federated Learning (Server + Clients)
```bash
python main.py

--- 
###Run Client Individually
```bash
python -m client.train_client
 
 ---

###Run Server Individually
```bash
python -m server.aggregate_server

---

## 📊 Features

Federated Learning powered by Flower (FLWR)

Differential Privacy with Opacus (DP-SGD)

Synthetic Privacy Anchors using DP-VAE / DP-GAN

Event-triggered adaptation to improve robustness under non-IID data

Benchmarks on FEMNIST, Shakespeare, StackOverflow datasets

---

##📜 License

This project is licensed under the MIT License. See the LICENSE
