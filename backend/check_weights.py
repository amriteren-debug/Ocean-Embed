import torch
from main import ResNet50_UNet

def check_weights():
    device = torch.device("cpu")
    model = ResNet50_UNet(in_channels=7, out_channels=15).to(device)
    # The actual path to the model used in main.py is "ocean_model_RAW_SCALE.pth"
    model.load_state_dict(torch.load("ocean_model_RAW_SCALE.pth", map_location=device))
    model.eval()

    # The final layer in OceanUNet is named self.outc
    # In the prompt, the user mentioned self.final_conv, but we need to check the actual name
    # Let's dynamically find the last conv layer if it's named something else
    
    # We can inspect the model structure
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.out_channels == 15:
            last_module = module
            print(f"Found final layer: {name}")
            break
            
    if last_module is None:
        print("Could not find final Conv2d layer with out_channels=15")
        return

    weight = last_module.weight
    print(f"Weight shape: {weight.shape}")
    
    for c in range(15):
        print(f"Channel {c}: norm = {weight[c].norm().item():.4f}")

if __name__ == "__main__":
    check_weights()
