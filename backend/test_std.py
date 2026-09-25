import torch
import segmentation_models_pytorch as smp
import numpy as np
import os
import sys

sys.path.append('c:/Ocean-Predictor(first)/backend')
from main import model as loaded_model, device

def test_std():
    print("--- TASK 1: STD of Model Output ---")
    input_tensor = torch.randn(1, 7, 160, 224).to(device)
    with torch.no_grad():
        out = loaded_model(input_tensor)
    
    # 1 (5m), 5 (50m), 14 (1000m)
    print(f"Index 1 (5m) std: {out[0, 1].std().item()}")
    print(f"Index 5 (50m) std: {out[0, 5].std().item()}")
    print(f"Index 14 (1000m) std: {out[0, 14].std().item()}")

if __name__ == "__main__":
    test_std()
