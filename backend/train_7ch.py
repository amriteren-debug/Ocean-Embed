"""
OceanEmbed Training Script (R6, R9)
=====================================
Training script for the production 7-channel ResNet50-UNet model that
reconstructs subsurface ocean temperature from surface satellite observations.

Architecture: ResNet50 encoder (pretrained on ImageNet) + UNet decoder
    Input:  7 channels (SST, SSS, SSH, U_cur, V_cur, U_wind, V_wind)
    Output: 15 channels (native GLORYS depth levels from 0.5m to 902m)

Training Target: GLORYS Global Ocean Physics Reanalysis
    Product: GLOBAL_MULTIYEAR_PHY_001_030
    DOI: https://doi.org/10.48670/moi-00021
    Dataset IDs:
        - cmems_mod_glo_phy_my_0.083deg_P1D-m (multi-year reanalysis)
    Resolution: 0.083 deg, daily
    Depth levels: 50 native levels from 0.49m to 5728m
    We select 15 levels spanning 0.5m-902m as training targets.

Training Data Preparation:
    1. Surface inputs (SST, SSS, SSH, currents) from GLORYS surface layer
    2. Wind from CMEMS NRT wind product (0.125 deg hourly -> daily mean)
    3. Target: GLORYS temperature at 15 selected depth levels
    4. All data regridded to native 0.083 deg for training
       (inference regrids to 0.25 deg via preprocessing.py)

Channel Order (must match inference in main.py):
    0: SST (thetao at depth[0])
    1: SSS (so at depth[0])
    2: SSH (zos)
    3: U_cur (uo at depth[0])
    4: V_cur (vo at depth[0])
    5: U_wind (eastward_wind, daily mean)
    6: V_wind (northward_wind, daily mean)

Usage:
    python train_7ch.py --data-dir training_data/ --epochs 100 --lr 1e-4

Note: The production checkpoint (BEST_ocean_model_1YR_7CH.pth) was trained
on 1 year of daily GLORYS data. Re-training requires significant compute
time and GLORYS data access via Copernicus Marine Service.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import xarray as xr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import scipy.ndimage

# ──────────────────────────────────────────────────────────────
# GLORYS depth levels used as training targets
# These are the 15 native GLORYS depth indices selected for the model
# ──────────────────────────────────────────────────────────────
NATIVE_DEPTHS_M = [
    0.494, 5.078, 9.573, 18.496, 29.445,
    47.374, 77.854, 92.326, 155.851, 186.126,
    318.127, 380.213, 541.089, 643.567, 902.339,
]


class GLORYS_7CH_Dataset(Dataset):
    """Dataset for training the 7-channel model on GLORYS reanalysis data.

    Expects pre-downloaded GLORYS NetCDF files in the data directory with
    variables: thetao, so, zos, uo, vo (from GLORYS), and optionally
    eastward_wind, northward_wind (from wind product).

    Data Source: GLORYS Global Ocean Physics Reanalysis (R9)
        DOI: https://doi.org/10.48670/moi-00021
    """

    def __init__(self, thetao_file, so_file, cur_file, zos_file, wind_file=None,
                 target_depth_indices=None):
        """
        Parameters
        ----------
        thetao_file : str — path to GLORYS thetao NetCDF (with depth dimension)
        so_file : str — path to GLORYS so NetCDF (with depth dimension)
        cur_file : str — path to GLORYS currents NetCDF (uo, vo with depth)
        zos_file : str — path to GLORYS SSH NetCDF (zos, no depth)
        wind_file : str or None — path to wind NetCDF (optional)
        target_depth_indices : list[int] or None — GLORYS depth indices for targets
        """
        print(f"Loading GLORYS training dataset...")
        self.ds_theta = xr.open_dataset(thetao_file)
        self.ds_so = xr.open_dataset(so_file)
        self.ds_cur = xr.open_dataset(cur_file)
        self.ds_zos = xr.open_dataset(zos_file)

        self.wind_available = False
        if wind_file and os.path.exists(wind_file):
            try:
                self.ds_wind = xr.open_dataset(wind_file)
                self.wind_available = True
            except Exception:
                pass

        self.n_times = len(self.ds_theta.time)
        print(f"  Found {self.n_times} time steps")

        # Determine target depth indices
        depths = self.ds_theta.depth.values
        if target_depth_indices is not None:
            self.target_indices = target_depth_indices
        else:
            # Find nearest indices for our target depths
            self.target_indices = []
            for d in NATIVE_DEPTHS_M:
                self.target_indices.append(int(np.abs(depths - d).argmin()))

        self.out_channels = len(self.target_indices)
        print(f"  Target depths ({self.out_channels} levels):")
        for i, idx in enumerate(self.target_indices):
            print(f"    [{i}] depth index {idx} = {depths[idx]:.3f}m")
            
        # Load target normalization stats
        stats_path = os.path.join(os.path.dirname(__file__), "target_depth_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.target_stats = json.load(f)
            print(f"  Loaded target normalization stats from {stats_path}")
        else:
            print("  WARNING: target_depth_stats.json not found! Targets will not be normalized.")
            self.target_stats = None

    def __len__(self):
        return self.n_times

    def __getitem__(self, idx):
        # ── Surface inputs (7 channels) ──
        sst = self.ds_theta['thetao'].isel(time=idx, depth=0).fillna(0).values
        sss = self.ds_so['so'].isel(time=idx, depth=0).fillna(0).values
        u_cur = self.ds_cur['uo'].isel(time=idx, depth=0).fillna(0).values
        v_cur = self.ds_cur['vo'].isel(time=idx, depth=0).fillna(0).values
        ssh = self.ds_zos['zos'].isel(time=idx).fillna(0).values

        if self.wind_available:
            # If wind has time dimension matching, use it
            try:
                u_wind = self.ds_wind['eastward_wind'].isel(time=idx).fillna(0).values
                v_wind = self.ds_wind['northward_wind'].isel(time=idx).fillna(0).values
            except Exception:
                u_wind = np.zeros_like(sst)
                v_wind = np.zeros_like(sst)
        else:
            u_wind = np.zeros_like(sst)
            v_wind = np.zeros_like(sst)

        # ── Coastal Infilling & Channel-wise Normalization ──
        raw_inputs = [sst, sss, ssh, u_cur, v_cur, u_wind, v_wind]
        norm_inputs = []
        for i, ch_data in enumerate(raw_inputs):
            mask = (ch_data != 0)
            if mask.any():
                mean = ch_data[mask].mean()
                std = ch_data[mask].std() + 1e-6
                # Nearest neighbor coastal infilling (extrapolation)
                indices = scipy.ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
                infilled = ch_data[tuple(indices)]
                # Standardize
                norm_inputs.append((infilled - mean) / std)
            else:
                norm_inputs.append(ch_data)
                
        input_tensor = torch.tensor(
            np.stack(norm_inputs, axis=0),
            dtype=torch.float32
        )

        # ── Target: temperature at selected depth levels ──
        target_layers = []
        for i, di in enumerate(self.target_indices):
            layer = self.ds_theta['thetao'].isel(time=idx, depth=di).fillna(0).values
            
            # Apply target normalization if stats are available
            if self.target_stats is not None:
                stat = self.target_stats[i]
                mask = (layer != 0)
                if mask.any():
                    # Only normalize valid ocean pixels, leave land as 0 for masking
                    norm_layer = np.zeros_like(layer)
                    norm_layer[mask] = (layer[mask] - stat["mean"]) / stat["std"]
                    layer = norm_layer
            
            target_layers.append(layer)

        target_tensor = torch.tensor(
            np.stack(target_layers, axis=0),
            dtype=torch.float32
        )

        return input_tensor, target_tensor

    def close(self):
        self.ds_theta.close()
        self.ds_so.close()
        self.ds_cur.close()
        self.ds_zos.close()
        if self.wind_available:
            self.ds_wind.close()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──
    dataset = GLORYS_7CH_Dataset(
        thetao_file=os.path.join(args.data_dir, args.thetao),
        so_file=os.path.join(args.data_dir, args.so),
        cur_file=os.path.join(args.data_dir, args.cur),
        zos_file=os.path.join(args.data_dir, args.zos),
        wind_file=os.path.join(args.data_dir, args.wind) if args.wind else None,
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # ── Model ──
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet", # Use pre-trained weights for from-scratch training
        in_channels=7,
        classes=dataset.out_channels,
    ).to(device)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_state_dict(
            torch.load(args.resume, map_location=device, weights_only=True)
        )

    # ── Optimizer & Loss ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Load Bathymetry Mask for Loss Calculation
    mask_3d = None
    mask_path = os.path.join(os.path.dirname(__file__), "bathymetry_mask.npy")
    if os.path.exists(mask_path):
        mask_np = np.load(mask_path)
        mask_3d = torch.tensor(mask_np, dtype=torch.bool).to(device)

    class MaskedDepthWeightedLoss(nn.Module):
        def __init__(self, mask_3d=None):
            super().__init__()
            self.mask_3d = mask_3d
            weights = torch.exp(-0.1 * torch.arange(15)).to(device)
            self.depth_weights = weights.view(1, 15, 1, 1)

        def forward(self, pred, target):
            loss = (pred - target) ** 2
            loss = loss * self.depth_weights
            if self.mask_3d is not None:
                # Mask out land boundaries to prevent coastal bleeding
                loss = loss[:, self.mask_3d]
                return loss.mean()
            else:
                mask = (target != 0)
                if mask.any():
                    loss = loss[mask]
                return loss.mean()

    criterion = MaskedDepthWeightedLoss(mask_3d=mask_3d)

    # ── Training Loop ──
    best_loss = float('inf')
    history = []

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Model: smp.Unet(resnet50, in=7, out={dataset.out_channels})")
    print(f"  LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Training target: GLORYS Reanalysis (DOI: 10.48670/moi-00021)")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(avg_loss)
        history.append({"epoch": epoch + 1, "loss": avg_loss})

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{args.epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.output)
            if (epoch + 1) % 10 == 0:
                print(f"    -> Saved best model (loss={best_loss:.6f})")

    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    print(f"Checkpoint saved to: {args.output}")

    # Save training history
    history_file = args.output.replace('.pth', '_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_file}")

    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train OceanEmbed 7-channel model on GLORYS data"
    )
    parser.add_argument("--data-dir", default="training_data",
                        help="Directory containing training NetCDF files")
    parser.add_argument("--thetao", default="thetao.nc",
                        help="GLORYS thetao filename (within data-dir)")
    parser.add_argument("--so", default="so.nc",
                        help="GLORYS salinity filename")
    parser.add_argument("--cur", default="cur.nc",
                        help="GLORYS currents filename")
    parser.add_argument("--zos", default="zos.nc",
                        help="GLORYS SSH filename")
    parser.add_argument("--wind", default=None,
                        help="Wind product filename (optional)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output", default="BEST_ocean_model_1YR_7CH.pth",
                        help="Output checkpoint path")
    args = parser.parse_args()

    train(args)
