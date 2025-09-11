# Event-Triggered Privacy Anchors (ETPA) for Federated Learning

## 📌 Project Overview
This project implements **Federated Learning with Differential Privacy (DP)** using two key ideas:

- **Privacy Anchors** → Synthetic samples generated via DP-VAE / DP-GAN  
- **Event-Triggered Adaptation** → Dynamically adjusting DP noise and regenerating anchors when client models diverge  

The aim is to achieve the right balance between **accuracy, privacy, and efficiency** in non-IID federated learning setups.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JoyBiswas1403/etpa-fl.git
cd etpa-fl

### 2. Setup Virtual Environment
```bash
python -m venv .venv
Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
Linux/Mac:

```bash
source .venv/bin/activate

### 3. Install Dependencies
```bash

pip install --upgrade pip
pip install -r requirements.txt

--- 

📂 Repository Structure

```bash
etpa-fl/
├── client/          # Client-side training logic
├── server/          # Server aggregation and coordination
├── utils/           # Data partitioning, evaluation, helper functions
├── anchors/         # Privacy Anchor generation (DP-VAE, DP-GAN)
├── requirements.txt # Python dependencies
├── README.md        # Project documentation
└── main.py          # Entry point

--- 

##🚀 Usage

###Run Full Federated Learning (Server + Clients)
```bash

python main.py
You should see:

```arduino

Starting Event-Triggered Privacy Anchors (ETPA) Simulation
Server is running and waiting for clients...
Client 1 is running...
Client 2 is running...
...

---

###📊 Results & Goals

- Balance between accuracy, privacy, and efficiency
- Evaluate under non-IID data distributions
- Compare against standard DP techniques

---