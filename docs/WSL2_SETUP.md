# WSL2 + SAM3 Setup Guide for Cortex-AI

This guide will set up WSL2 with Ubuntu, CUDA, and SAM3 for full triton-accelerated inference.

## Prerequisites

- Windows 10 (Build 19041+) or Windows 11
- NVIDIA GPU with CUDA support (GTX 1650 ✅)
- Latest NVIDIA Windows drivers installed

---

## Step 1: Install Ubuntu on WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install -d Ubuntu-22.04
```

After installation, set username and password when prompted.

---

## Step 2: Update Ubuntu & Install Dependencies

```bash
# In WSL2 Ubuntu terminal
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential git curl wget python3-pip python3-venv

# Install CUDA toolkit for WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1
```

---

## Step 3: Set Up Python Environment

```bash
# Create project directory (accessible from Windows)
mkdir -p ~/cortex-ai
cd ~/cortex-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

---

## Step 4: Install SAM3

```bash
# Clone SAM3 repository
git clone https://github.com/facebookresearch/sam3.git
cd sam3

# Install SAM3 and dependencies
pip install -e .

# Install triton (Linux only - this is why we need WSL!)
pip install triton

# Login to Hugging Face for model access
pip install huggingface_hub
huggingface-cli login
```

---

## Step 5: Install Cortex-AI Backend

```bash
# Go back to project directory
cd ~/cortex-ai

# Clone your Cortex-AI project (or copy from Windows)
# Option A: Clone from GitHub
git clone https://github.com/DilipReddy57/Cortex-AI.git
cd Cortex-AI

# Option B: Copy from Windows path
# cp -r /mnt/d/project/data\ annotaion\ using\ sam3/* .

# Install requirements
pip install -r requirements.txt
pip install -e backend/sam3
```

---

## Step 6: Test SAM3

```bash
python -c "
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

print('Loading SAM3 model...')
model = build_sam3_image_model()
processor = Sam3Processor(model)
print('✅ SAM3 loaded successfully!')
print(f'Device: {next(model.parameters()).device}')
"
```

---

## Step 7: Run Cortex-AI Backend

```bash
cd ~/cortex-ai/Cortex-AI

# Start the FastAPI server
python -m backend.main

# Server will be available at http://localhost:8000
# This is accessible from Windows browser!
```

---

## Accessing WSL from Windows

### File Access

- WSL files: `\\wsl$\Ubuntu-22.04\home\<username>\cortex-ai`
- Windows files from WSL: `/mnt/c/`, `/mnt/d/`, etc.

### Network Access

- WSL services on `localhost` are accessible from Windows
- Your frontend at `http://localhost:5173` can call backend at `http://localhost:8000`

---

## Quick Commands

```bash
# Start WSL
wsl

# Enter project
cd ~/cortex-ai/Cortex-AI && source ../venv/bin/activate

# Run backend
python -m backend.main

# Check GPU
nvidia-smi
```

---

## Troubleshooting

### CUDA not detected

```bash
# Add CUDA to path
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Out of memory on GTX 1650 (4GB)

```bash
# Use smaller batch size or enable memory optimization in config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Permission denied

```bash
sudo chown -R $USER:$USER ~/cortex-ai
```
