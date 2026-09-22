import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import time
import argparse
import os
import json
from pathlib import Path

from trainer import *
from data import *
from plotter import Plotter

"""
Parse command-line arguments for training and inference.

Returns
-------
argparse.Namespace
    Parsed command-line arguments.
"""
def parse_args():
    parser = argparse.ArgumentParser(description="MLP Training and Inference")

    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto", help="Choose device: cpu, gpu, or auto (default: auto)")
    parser.add_argument("inputs", type=str, nargs="*", default=["clustersTracks.csv"], help="One or more input CSV files")
    parser.add_argument("--max_epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for DataLoader")
    parser.add_argument("--outdir", type=str, default="outputs/local", help="Directory to save models and plots")
    parser.add_argument("--end_name", type=str, default="", help="Optional suffix to append to output files (default: none)")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate for the MLP model (default: 0.1)")
    parser.add_argument("--no_train", action="store_true", help="Skip training and only run inference using a saved model")
    parser.add_argument("--enable_progress_bar", action="store_true", help="Enable progress bar during training (default: disabled)")
    parser.add_argument("--model_file", type=str, default=None, help="Path to TorchScript model (.pt). If omitted, use the default model.")
    parser.add_argument("--norm_file", type=str, default=None, help="Path to normalization statistics (norm_stats.json). If omitted, use the default normalization.")
    return parser.parse_args()


class TrackDataset(Dataset):
    def __init__(self, features, targets, normalize=True, stats=None):
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.normalize = normalize
        self.input_dim = self.features.shape[1]
        self.target_dim = self.targets.shape[1]

        # Normalization statistics
        if stats is None:
            self.stats = {}
            for i in range(self.input_dim):
                self.stats[f"in_mean_{i}"] = 0.0
                self.stats[f"in_std_{i}"] = 1.0
            for i in range(self.target_dim):
                self.stats[f"out_mean_{i}"] = 0.0
                self.stats[f"out_std_{i}"] = 1.0
        else:
            self.stats = stats

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].copy()
        y = self.targets[idx].copy()

        if self.normalize:
            for layer in range(9):
                if layer < 3:
                    start = layer * 5
                    # Six endpoint coordinates
                    for j in range(4):
                        i = start + j
                        x[i] = (x[i] - self.stats[f"in_mean_{i}"]) / self.stats[f"in_std_{i}"]

                    # Mask
                    mask_index = start + 4

                    if x[mask_index] == 0:
                        x[start:start + 4] = 0.0
                else:
                    start = 3*5 + (layer-3) * 4
                    # Six endpoint coordinates
                    for j in range(3):
                        i = start + j
                        x[i] = (x[i] - self.stats[f"in_mean_{i}"]) / self.stats[f"in_std_{i}"]

                    # Mask
                    mask_index = start + 3

                    if x[mask_index] == 0:
                        x[start:start + 3] = 0.0

            # Normalize track state
            # [d0, phi0, kappa, z0, tandip]
            for i in range(self.target_dim):
                y_mean = self.stats[f"out_mean_{i}"]
                y_std = self.stats[f"out_std_{i}"]
                y[i] = (y[i] - y_mean) / y_std

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y,dtype=torch.float32)
        )


def main():

    args = parse_args()

    outDir = args.outdir
    maxEpochs = args.max_epochs
    batchSize = args.batch_size
    end_name = args.end_name

    doTraining = not args.no_train

    os.makedirs(
        outDir,
        exist_ok=True
    )

    # Load data
    print("\n\nLoading data...")

    startT_data = time.time()

    features_all = []
    targets_all = []

    for fname in args.inputs:
        print(f"Loading data from {fname} ...")
        features, targets = read_track_state_data(fname)
        features_all.append(features)
        targets_all.append(targets)

    # Concatenate all files
    features_all = np.concatenate(features_all, axis=0).astype(np.float32)
    targets_all = np.concatenate(targets_all,axis=0).astype(np.float32)
    feature_dim = features_all.shape[1]
    target_dim = targets_all.shape[1]

    # Check dimensions
    print(f"\nLoaded {len(targets_all)} samples")
    print(f"Feature dimension: {feature_dim}")
    print(f"Target dimension : {target_dim}")

    # Train / validation split
    train_size = int(0.8 * len(features_all))
    val_size = (len(features_all) - train_size)

    indices = np.random.RandomState(42).permutation(len(features_all))

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_features = features_all[train_idx]
    train_targets = targets_all[train_idx]

    val_features = features_all[val_idx]
    val_targets = targets_all[val_idx]

    # Paths
    norm_stats_out_path = os.path.join(outDir, "norm_stats.json")
    BASE_DIR = Path(__file__).resolve().parent
    NETS_DIR = BASE_DIR / "nets"

    # Load or compute normalization statistics
    if args.no_train:
        stats_path = (
            Path(args.norm_file).expanduser().resolve()
            if args.norm_file is not None
            else NETS_DIR / "norm_stats.json"
        )
        print(f"Loading normalization stats from: {stats_path}")

        if not stats_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
        with open(stats_path,"r") as f:
            norm_stats = json.load(f)
        print("Loaded normalization statistics.")
    else:
        print("\n=== Computing normalization statistics ===")
        norm_stats = {}

        # ======================================================
        # Input feature statistics
        # Only samples with mask == 1 are used for calculating
        # coordinate mean/std.
        # ======================================================
        for layer in range(9):
            if layer < 3:
                start = layer * 5
                mask_index = start + 4

                # Select valid measurements
                valid_mask = (train_features[:, mask_index] == 1)
                n_valid = np.sum(valid_mask)
                print(f"Layer {layer + 1:2d}: {n_valid} valid / {len(train_features)} total ({100.0 * n_valid / len(train_features):.2f}%)")

                # Calculate statistics for six endpoint coordinates using ONLY valid measurements.
                for j in range(4):
                    i = start + j
                    valid_values = train_features[valid_mask,i]
                    if len(valid_values) == 0:
                        raise ValueError(f"Layer {layer + 1}, feature {j}: no valid measurements found.")
                    mean = valid_values.mean()
                    std = valid_values.std()

                    # Protect against zero standard deviation
                    if std < 1.0e-8:
                        std = 1.0

                    norm_stats[f"in_mean_{i}"] = float(mean)
                    norm_stats[f"in_std_{i}"] = float(std)
            else:
                start = 3*5 + (layer-3) * 4
                mask_index = start + 3

                # Select valid measurements
                valid_mask = (train_features[:, mask_index] == 1)
                n_valid = np.sum(valid_mask)
                print(
                    f"Layer {layer + 1:2d}: {n_valid} valid / {len(train_features)} total ({100.0 * n_valid / len(train_features):.2f}%)")

                # Calculate statistics for six endpoint coordinates using ONLY valid measurements.
                for j in range(3):
                    i = start + j
                    valid_values = train_features[valid_mask, i]
                    if len(valid_values) == 0:
                        raise ValueError(f"Layer {layer + 1}, feature {j}: no valid measurements found.")
                    mean = valid_values.mean()
                    std = valid_values.std()

                    # Protect against zero standard deviation
                    if std < 1.0e-8:
                        std = 1.0

                    norm_stats[f"in_mean_{i}"] = float(mean)
                    norm_stats[f"in_std_{i}"] = float(std)

            # Mask itself is NOT normalized
            norm_stats[f"in_mean_{mask_index}"] = 0.0
            norm_stats[f"in_std_{mask_index}"] = 1.0

        # Target statistics
        # [d0, phi0, kappa, z0, tandip]
        for i in range(target_dim):
            mean = train_targets[:, i].mean()
            std = train_targets[:, i].std()

            if std < 1.0e-8:
                std = 1.0

            norm_stats[f"out_mean_{i}"] = float(mean)
            norm_stats[f"out_std_{i}"] = float(std)

        # Save normalization statistics
        with open(norm_stats_out_path, "w") as f:
            json.dump(norm_stats, f, indent=2)

        print("\nSaved normalization stats to:")
        print(norm_stats_out_path)
        print("=========================================\n")

    # Dataset
    train_set = TrackDataset(train_features, train_targets, normalize=True, stats=norm_stats)
    val_set = TrackDataset(val_features, val_targets, normalize=True, stats=norm_stats)

    print(f"\nTrain size: {train_size}")
    print(f"Validation size: {val_size}")

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchSize, shuffle=False)

    # Check sample
    x_cont, targets = next(iter(train_loader))

    print("x_cont shape:", x_cont.shape)
    print("target shape:", targets.shape)

    endT_data = time.time()

    print(f"\nLoading data took {endT_data - startT_data:.2f} s\n")

    # Plotter
    plotter = Plotter(print_dir=outDir, end_name=end_name)

    # Model
    model = TrackStateMLP(
        input_dim=feature_dim,
        state_dim=target_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        dropout=args.dropout,
        phi_mean=norm_stats[f"out_mean_{1}"],
        phi_std=norm_stats[f"out_std_{1}"]
    )

    # Loss tracker
    loss_tracker = LossTracker()

    # Training
    if doTraining:
        # Device selection
        if args.device == "cpu":
            accelerator, devices = "cpu", 1
        elif args.device == "gpu":
            if torch.cuda.is_available():
                accelerator, devices = "gpu", 1
            else:
                print("GPU requested but not available. Falling back to CPU.")
                accelerator, devices = "cpu", 1
        elif args.device == "auto":
            if torch.cuda.is_available():
                accelerator, devices = "gpu", "auto"
            else:
                accelerator, devices = "cpu", 1
        else:
            raise ValueError(f"Unknown device option: {args.device}")
        print(f"Using accelerator={accelerator}, devices={devices}")

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy="auto",
            max_epochs=maxEpochs,
            enable_progress_bar=(args.enable_progress_bar),
            log_every_n_steps=100,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            gradient_clip_val=1.0,
            logger=False,
            callbacks=[loss_tracker]
        )

        print("\n\nTraining...")

        startT_train = time.time()

        trainer.fit(model, train_loader, val_loader)

        endT_train = time.time()

        print(f"Training took {(endT_train - startT_train) / 60:.2f} minutes\n\n")

        # Plot training loss
        plotter.plotTrainLoss(loss_tracker)

        # Save TorchScript model
        model.to("cpu")
        scripted_model = torch.jit.script(model)
        model_path = (f"{outDir}/track_state_mlp_{end_name}.pt")
        scripted_model.save(model_path)
        print(f"Saved model to: {model_path}")

    # Load model
    BASE_DIR = Path(__file__).resolve().parent

    if doTraining:
        model_file = (Path(outDir) / f"track_state_mlp_{end_name}.pt")
    else:
        model_file = (
            Path(args.model_file).expanduser().resolve()
            if args.model_file is not None
            else BASE_DIR / "nets" / "track_state_mlp_default.pt"
        )

    model_file = model_file.resolve()

    print( "Loading model from:", model_file)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = torch.jit.load(model_file)

    model.eval()

    # Inference
    print("\nRunning inference...")

    startT_test = time.time()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for feature, target in val_loader:
            pred = model(feature)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    endT_test = time.time()

    print(f"\nInference with {val_size} samples took {endT_test - startT_test:.2f} s\n")

    # Concatenate outputs
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    print("Predictions shape :", all_preds.shape)
    print("Targets shape     :", all_targets.shape)

    # Metrics in normalized space
    mse = torch.mean((all_preds - all_targets) ** 2)
    mae = torch.mean(torch.abs(all_preds - all_targets))
    print(f"MSE = {mse:.6f}")
    print(f"MAE = {mae:.6f}")

    # Denormalize predictions and targets
    out_mean = torch.tensor(
        [
            norm_stats[f"out_mean_{i}"]
            for i in range(target_dim)
        ],
        dtype=torch.float32
    )

    out_std = torch.tensor(
        [
            norm_stats[f"out_std_{i}"]
            for i in range(target_dim)
        ],
        dtype=torch.float32
    )

    all_preds_denorm = (all_preds * out_std + out_mean)
    all_targets_denorm = (all_targets * out_std + out_mean)

    # Generate comparison plots
    plotter.plot_residuals(all_preds_denorm, all_targets_denorm)
    plotter.plot_pred_target(all_preds_denorm, all_targets_denorm)

    # Target statistics
    state_names = ["d0", "phi0", "kappa", "z0", "tandip"]

    print("\nTarget statistics")

    for i, name in enumerate(state_names):
        print(
            f"{name:7s}: mean={all_targets_denorm[:, i].mean():.6f} std={all_targets_denorm[:, i].std():.6f}")

    # Prediction statistics
    print("\nPrediction statistics")
    for i, name in enumerate(state_names):
        print(f"{name:7s}: mean={all_preds_denorm[:, i].mean():.6f} std={all_preds_denorm[:, i].std():.6f}")

    # Sample predictions
    print("\nSample predictions:")

    for i in range(min(target_dim, all_preds_denorm.size(0))):
        print(f"\nEvent {i}")
        for j, name in enumerate(state_names):
            print(f"{name:7s}  pred={all_preds_denorm[i, j]:12.6f}   target={all_targets_denorm[i, j]:12.6f}")


if __name__ == "__main__":
    main()