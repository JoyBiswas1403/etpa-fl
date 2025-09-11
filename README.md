# Event-Triggered Privacy Anchors (ETPA) for Federated Learning

## 📌 Project Overview

This project implements **Federated Learning with Differential Privacy (DP)** using two key concepts:

- **Privacy Anchors**: Synthetic samples generated via DP-VAE / DP-GAN  
- **Event-Triggered Adaptation**: Dynamically adjusts DP noise and regenerates anchors when client models diverge  

The aim is to balance **accuracy, privacy, and efficiency** in non-IID federated learning setups.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JoyBiswas1403/etpa-fl.git
cd etpa-fl
```

### 2. Set Up Virtual Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

- **Windows (PowerShell):**
    ```bash
    .venv\Scripts\Activate.ps1
    ```
- **Windows (Command Prompt):**
    ```cmd
    .venv\Scripts\activate.bat
    ```
- **Linux/Mac:**
    ```bash
    source .venv/bin/activate
    ```

### 3. Install Dependencies

Upgrade pip and install required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Troubleshooting Installation

*   **Error on Windows about running scripts:** You may need to set the execution policy first:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
*   **Opacus/CUDA errors:** If you don't have a GPU, ensure you are installing the CPU-only version of PyTorch first. Check the [Opacus installation guide](https://opacus.ai/) for the correct `pip install` command.

## 📂 Repository Structure

```
etpa-fl/
├── client/          # Client-side training logic
├── server/          # Server aggregation and coordination
├── utils/           # Data partitioning, evaluation, helper functions
├── anchors/         # Privacy Anchor generation (DP-VAE, DP-GAN)
├── requirements.txt # Python dependencies
├── README.md        # Project documentation
└── main.py          # Entry point
```

---

## 🚀 Usage

To run the full Federated Learning simulation (server + clients):

```bash
python main.py
```

You should see output like:

```text
Starting Event-Triggered Privacy Anchors (ETPA) Simulation
Server is running and waiting for clients...
Client 1 is running...
Client 2 is running...
...
```

---

## 📊 Results & Goals

- Achieve a balance between accuracy, privacy, and efficiency
- Evaluate performance under non-IID data distributions
- Compare results against standard DP techniques

---

## 🤝 Contributing

Contributions are welcome! Please open issues and pull requests to suggest improvements or report bugs.

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


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