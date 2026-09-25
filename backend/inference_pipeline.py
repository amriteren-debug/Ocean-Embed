import os
import json
import glob
import torch
import numpy as np
import segmentation_models_pytorch as smp

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Initializes the U-Net model (ResNet50, 7-channel input, 15-class output)
    and loads the trained checkpoint.
    """
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=7,
        classes=15,
    )
    
    # Load state dict strictly mapping to the provided device
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def load_stats(json_path: str, device: torch.device):
    """
    Parses the target_depth_stats.json dynamically to extract means and stds.
    
    Returns:
        means_tensor (torch.Tensor): Shape (15, 1, 1)
        stds_tensor  (torch.Tensor): Shape (15, 1, 1)
    """
    with open(json_path, 'r') as f:
        stats = json.load(f)
        
    # Sort by layer index to ensure alignment (0 to 14)
    stats = sorted(stats, key=lambda x: x["layer"])
    
    means = [item["mean"] for item in stats]
    stds = [item["std"] for item in stats]
    
    means_tensor = torch.tensor(means, dtype=torch.float32, device=device).view(15, 1, 1)
    stds_tensor = torch.tensor(stds, dtype=torch.float32, device=device).view(15, 1, 1)
    
    return means_tensor, stds_tensor

def load_test_data(data_dir: str):
    """
    Dynamically finds the first .pt file in the training_data directory.
    Extracts the input and target tensors safely.
    """
    # Recursively search for .pt files
    pt_files = glob.glob(os.path.join(data_dir, "**", "*.pt"), recursive=True)
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")
        
    file_path = pt_files[0]
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    
    # Safely extract wrapped datasets (dict, tuple, or raw tensor)
    if isinstance(data, dict):
        inputs = data.get("inputs", data.get("input"))
        targets = data.get("targets", data.get("target"))
    elif isinstance(data, (tuple, list)):
        inputs = data[0]
        targets = data[1] if len(data) > 1 else None
    else:
        inputs = data
        targets = None
        
    # Enforce batch dimension for inference (B, C, H, W)
    if inputs is not None and inputs.dim() == 3:
        inputs = inputs.unsqueeze(0)
    if targets is not None:
        # Some legacy tensors contain 36 layers; we only predict the first 15 native GLORYS depths
        if targets.dim() == 3:
            targets = targets[:15, :, :]
            targets = targets.unsqueeze(0)
        elif targets.dim() == 4:
            targets = targets[:, :15, :, :]
        
        
    return inputs, targets, file_path

def predict(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor, 
            means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    """
    Executes standard inference and mathematically exact post-processing:
      a) Model inference (no_grad)
      b) Un-normalizes exactly against the target statistics
      c) Applies the logical land mask based on ground-truth targets (target == 0)
    """
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    # a) Inference
    with torch.no_grad():
        preds = model(inputs)
        
    # Assume single batch item processing for simplicity
    preds = preds.squeeze(0)  # Shape: (15, H, W)
    
    # b) Un-normalize: T_true = (T_pred * std) + mean
    preds_unnorm = (preds * stds) + means
    
    # c) Apply Logical Land Mask
    # Since the model was NOT penalized for land (targets == 0) via MaskedDepthWeightedLoss,
    # the predictions there are unconstrained. We mathematically force them to 0.0.
    if targets is not None:
        targets = targets.to(device).squeeze(0)  # Shape: (15, H, W)
        
        # Where the true target is exactly 0.0 (Land/Missing), force prediction to 0.0
        land_mask = (targets == 0.0)
        preds_unnorm[land_mask] = 0.0
        
    return preds_unnorm

def main():
    # Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "model_1YR_7CH_BEST_OVERALL.pth")
    stats_path = os.path.join(base_dir, "models", "target_depth_stats.json")
    data_dir = os.path.join(base_dir, "training_data")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Inference Pipeline on: {device}")
    
    # 1. Load Normalization Stats
    print("-> Loading target depth stats...")
    means, stds = load_stats(stats_path, device)
    
    # 2. Load Model Checkpoint
    print("-> Loading U-Net (ResNet50) model checkpoint...")
    model = load_model(model_path, device)
    
    # 3. Load Sample Test Data
    print(f"-> Scanning for test data in: {data_dir}")
    inputs, targets, file_path = load_test_data(data_dir)
    print(f"   [Loaded]: {os.path.basename(file_path)}")
    print(f"   [Inputs Shape]:  {inputs.shape}")
    if targets is not None:
        print(f"   [Targets Shape]: {targets.shape}")
    
    # 4. Predict & Post-Process
    print("-> Running inference and applying mathematical post-processing...")
    final_preds = predict(model, inputs, targets, means, stds)
    
    print(f"-> [SUCCESS] Final output tensor shape: {final_preds.shape}")
    print("-> Tensor is ready for downstream API serialization.")
    
    # Return as raw numpy array for demonstration/downstream usage
    return final_preds.cpu().numpy()

if __name__ == "__main__":
    _ = main()
