#!/usr/bin/env python3
"""
GPT Fine-tuning Environment Setup
This script sets up the environment for fine-tuning open-source GPT models.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for GPT fine-tuning."""
    
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "wandb",  # for experiment tracking
        "peft>=0.4.0",  # for parameter-efficient fine-tuning
        "bitsandbytes",  # for quantization
        "scipy",
        "scikit-learn",
        "tqdm",
        "numpy",
        "pandas"
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("\nEnvironment setup complete!")

def check_gpu():
    """Check if CUDA/MPS is available for training."""
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available with {torch.cuda.device_count()} GPU(s)")
            print(f"  Primary GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) available")
        else:
            print("⚠ Only CPU available - training will be slow")
            
    except ImportError:
        print("⚠ PyTorch not installed yet")

if __name__ == "__main__":
    print("=== GPT Fine-tuning Environment Setup ===\n")
    check_gpu()
    print()
    install_requirements()
