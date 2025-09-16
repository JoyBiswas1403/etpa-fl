pip install --upgrade pip
pip install -r requirements.txt
# Event-Triggered Privacy Anchors (ETPA) for Federated Learning

## 📌 Project Overview
This project implements **Federated Learning with Differential Privacy (DP)** using two key ideas:

- **Privacy Anchors** → Synthetic samples generated via DP-VAE / DP-GAN  
- **Event-Triggered Adaptation** → Dynamically adjusting DP noise and regenerating anchors when client models diverge  

<<<<<<< HEAD
The aim is to achieve the right balance between **accuracy, privacy, and efficiency** in non-IID federated learning setups.
=======
- **Privacy Anchors**: Synthetic samples generated via DP-VAE / DP-GAN
- **Event-Triggered Adaptation**: Dynamically adjusts DP noise and regenerates anchors when client models diverge

The aim is to balance **accuracy, privacy, and efficiency** in non-IID federated learning setups.
>>>>>>> 53322fef5a934a379f580fc7bd3ed60a5a8d40f6

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/JoyBiswas1403/etpa-fl.git
cd etpa-fl
<<<<<<< HEAD
=======
```

### 2. Setup Virtual Environment
>>>>>>> 53322fef5a934a379f580fc7bd3ed60a5a8d40f6

### 2. Setup Virtual Environment
```bash
python -m venv .venv
<<<<<<< HEAD
Windows (PowerShell):
=======
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies
>>>>>>> 53322fef5a934a379f580fc7bd3ed60a5a8d40f6

```bash
.venv\Scripts\Activate.ps1
Linux/Mac:

```bash
source .venv/bin/activate

### 3. Install Dependencies
```bash

pip install --upgrade pip
pip install -r requirements.txt

### 4. Generate Privacy Anchors
Before running experiments, generate anchors:
```bash
python anchors/generate_anchors.py --output anchors/test_anchors.pt

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

<<<<<<< HEAD
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
=======
## 🚀 Usage

### 1. Generate Anchors (if not already present)
```bash
python etpa-fl/anchors/generate_anchors.py --output anchors/test_anchors.pt
```

### 2. Start the Server (in one terminal)
```bash
python etpa-fl/server/server.py
```
>>>>>>> 53322fef5a934a379f580fc7bd3ed60a5a8d40f6

### 3. Run Clients (in separate terminals)
```bash
python etpa-fl/client/train_client.py --client_id 0
python etpa-fl/client/train_client.py --client_id 1
python etpa-fl/client/train_client.py --client_id 2
```

### 4. (Optional) Visualize Anchors
```bash
python experiments/test_anchors.py --anchors anchors/test_anchors.pt --visualize --num-samples 36
```

### 5. (Optional) Run the Main Test Script
```bash
python etpa-fl/main.py
```

---

## 📝 Workflow Summary

- Clients train on private + anchor data with weighted loss.
- Divergence is measured and (simulated) sent to the server.
- Server triggers events if divergence exceeds threshold and simulates anchor regeneration.
- Anchor visualization shows the quality of generated anchors.

---

<<<<<<< HEAD
###📊 Results & Goals
=======
## 📊 Results & Goals
>>>>>>> 53322fef5a934a379f580fc7bd3ed60a5a8d40f6

- Balance between accuracy, privacy, and efficiency
- Evaluate under non-IID data distributions
- Compare against standard DP techniques

---