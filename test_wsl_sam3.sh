#!/bin/bash
# Test SAM3 installation in WSL2

source ~/cortex-ai/venv/bin/activate
python << 'EOF'
import sys
print("Python:", sys.version)

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"PyTorch ERROR: {e}")

try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError as e:
    print(f"Triton ERROR: {e}")

try:
    from sam3.model_builder import build_sam3_image_model
    print("SAM3 model_builder: OK")
except ImportError as e:
    print(f"SAM3 ERROR: {e}")

print("\n✅ All checks complete!")
EOF
