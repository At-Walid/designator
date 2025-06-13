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

## ⚠ Requirements

- **Windows OS**  
  The tool requires Windows **because the simulator is built using Unreal Engine** and is currently only compiled as a Windows executable (`Mars.exe`).

- **GPU Support**  
  The tool requires a **CUDA-compatible GPU** for inference using PyTorch and MXNet-based models.

---

## 🧪 How to Use

### 1. 📦 Download the package

Download the tool from Zenodo.

---

### 2. 🛠️ Create and activate a new Python 3.10 virtual environment

Open Command Prompt and run:

```bash
# Create a new virtual environment with Python 3.10
python -m venv venv_py310

# Activate the environment
venv_py310\Scripts\activate

# Upgrade pip to ensure compatibility
python -m pip install --upgrade pip

# Upgrade pip to ensure compatibility
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

# Install AirSim manually
cd DESIGNATE_Tool_PY3.10\AirSim\PythonClient
python setup.py install

# Launch the GUI tool
cd ../../
python .\app\run.py
```

Make sure python and pip versions are both 3.10


## 📌 Limitations

- ❌ Not supported on Linux/macOS (due to `Mars.exe` dependency)
- ❌ Requires discrete GPU (NVIDIA CUDA support)
- ❌ Simulator must be manually recompiled for non-Windows targets (if needed)

---

## 📧 Contact

For academic use, collaboration, or simulator recompilation requests, contact:

**Mohammed Attaoui**  
[University of Luxembourg]  
Email: mohammed.attaoui@uni.lu