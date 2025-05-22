Copyright (c) 2025 University of Luxembourg.

# DESIGNATOR: : a Tool for Automated GAN-based Testing of DNNs in Martian Environments
This project provides a GUI-based testing framework for evaluating the robustness and accuracy of Deep Neural Networks (DNNs) using a high-fidelity Mars simulator built with Unreal Engine. The simulator is embedded directly into the GUI, allowing researchers and practitioners to visually observe how segmentation models perform under a variety of simulated terrain and lighting conditions.

---

## 🖥 Features

- GUI with simulator control and real-time visual feedback
- Multi-objective genetic search for input generation
- Integration of GAN (Pix2PixHD) for realism-aware image transformation
- DNN prediction visualization and mask comparison
- Automatic fitness scoring using IoU and feature-based diversity
- Search variants (DESIGNATE, DESIGNATE_SINGLE, DESIGNATE_PIXEL, etc.)

---


## Authors: 
-   Mohammed Oualid Attaoui - mohammed.attaoui@uni.lu
-   Fabrizio Pastore - fabrizio.pastore@uni.lu


## 📷 GUI Overview

- Simulated Image (raw synthetic from simulator)
- Realistic Image (GAN-enhanced)
- Label Image (ground truth segmentation)
- Prediction Image (model output)
- Fitness scores and archive size display

---

## This package includes:
- ✅ A pre-built Docker image of the DESIGNATE GUI tool
- 🪐 The Unreal-based Mars simulator (`MarsSim/`)
- ⚙️ A one-click script to launch the tool via Docker
- 🧩 Instructions for manual setup (without Docker)

## 💻 Requirements

### For Docker-based use:
- **Windows 10/11**
- **[Docker Desktop](https://www.docker.com/products/docker-desktop)** installed
- **GPU support (CUDA 11.3)** for running PyTorch

### For Manual installation:
- **Windows 10/11**
- **Python 3.7** (strictly required)
- **CUDA-compatible GPU**
- **All dependencies listed in `requirements.txt`**

---

## 🚀 Option 1: Use Docker (Recommended)

### ✅ Steps

1. **Download and unzip this folder from: https://doi.org/10.5281/zenodo.15449510**
2. unzip MarsSim.zip -d app/
3. Open **Command Prompt** in the folder
4. Run:

```bat
run_docker.bat
```

This will:
- Load the Docker image from `designate-gui.tar`
- Start the GUI
- Launch and control the simulator from inside the tool

---

## 🧩 Option 2: Run Without Docker (Manual Setup)

### ⚠️ This requires Python 3.7 and a compatible GPU

### 1. Set up a virtual environment

```bash
python3.7 -m venv venv
venv\Scripts\activate   # On Windows
```

### 2. Install the dependencies

```bash
pip install --upgrade pip

# Pre-install to avoid AirSim setup crash
pip install numpy==1.21.6 msgpack-rpc-python==0.4.1

# Install PyTorch with CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Then the rest
pip install -r requirements.txt
```

### 3. Run the tool
Download: https://doi.org/10.5281/zenodo.15491076
unzip MarsSim.zip -d app/
```bash
python app/run.py
```

The GUI will open. 

## 📁 Folder Contents

- `designate-gui.tar` → Prebuilt Docker image
- `MarsSim/` → Unreal engine-based simulator
- `run_docker.bat` → Launches the Docker-based tool
- `requirements.txt` → For manual setup
- `app/` → Source code, GUI, models, simulator launcher
- `README.md` → This file

---


## ⚠️ Notes

- This tool only works on **Windows** (due to the Unreal simulator)
- Manual installation requires **Python 3.7**
- The simulator is launched **internally by the GUI**


## 📧 Contact

For academic use, collaboration, or simulator recompilation requests, contact:

**Mohammed Attaoui**  
[University of Luxembourg]  
Email: mohammed.attaoui@uni.lu