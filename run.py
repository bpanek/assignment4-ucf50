"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model parameters,
and training hyperparameters. Depending on the selected mode ('train' or 'eval'), it performs
the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model (e.g., LRCN) and defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop and saves the best model weights.

- Eval mode:
    - Loads the pre-generated dataset splits.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set and prints overall test accuracy.
    
The module also includes a helper function for parsing command-line arguments.
"""

import os
import argparse
import json
from pathlib import Path
import wandb

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import numpy as np

from video_datasets import VideoDataset, load_dataset, dataset_split
from utils import transform_stats, compose_data_transforms, train_val_dloaders, test_dloaders
from models import LRCN
from train import train
from video_windows import FrozenVideoEncoder, VideoWindowDataset, encode_split
from test import test, get_test_report, get_confusion_matrix

def args_parser():
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
        
    Arguments include:
        -fd/--frame_dir: Directory for storing video frames.
        -trs/--train_size: Proportion of data to use for training (default 0.7).
        -tss/--test_size: Proportion of data to use for testing (default 0.1).
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type, either '3dcnn' or 'lrcn' (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        -p/--pretrained: Whether to use a pretrained CNN backbone (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Number of training epochs (default 30).
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames')
    parser.add_argument("--split_manifest", type=Path, help="Path to a saved split JSON")
    parser.add_argument('-trs', '--train_size', type=float, default=0.7, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int, help='Number of classes for the classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', help='3D CNN or LRCN', default='lrcn')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34', help='2D CNN backbone - options: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument("--temporal_pool", choices=("last", "mean"), default="last",help="How to combine the LSTM outputs across frames.")
    parser.add_argument('-p', '--pretrained', help='Use pretrained 2D CNN backbone', default=True)
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')

    parser.add_argument('-m', '--mode', type=str, default='train', help="Either 'train' or 'eval'", required=True)
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')

    args = parser.parse_args()
    return args

def load_manifest_splits(manifest_path, frame_dir, n_classes):
    """Load saved splits and resolve clip paths against the frame directory."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["class_labels"]) != n_classes:
        raise ValueError("Manifest class count does not match --n_classes.")

    return tuple(
        [(str(Path(frame_dir) / path), int(label))
         for path, label in manifest["splits"][name]]
        for name in ("train", "validation", "test")
    )

def build_window_loaders(encoder, manifest_path, video_root, device, num_workers):
    """Encode the fixed train and validation splits in manifest order."""
    train_samples, val_samples, _ = load_manifest_splits(manifest_path, ".", 50)
    train_data = encode_split(
        encoder, VideoWindowDataset(train_samples, video_root), device, num_workers
    )
    val_data = encode_split(
        encoder, VideoWindowDataset(val_samples, video_root), device, num_workers
    )
    return {
        "train": DataLoader(train_data, batch_size=128, shuffle=True),
        "val": DataLoader(val_data, batch_size=128),
    }


def train_window_lrcn(video_root, manifest_path, output_dir, num_workers=2):
    """Train the LRCN on frozen R(2+1)D-18 window embeddings.

    Encode the training and validation clips from the fixed split manifest.
    Train only the LRCN, log loss and accuracy to Weights & Biases, and
    select the checkpoint with the highest validation accuracy.

    Args:
        video_root: Directory containing the UCF50 action-class folders.
        manifest_path: Path to the JSON file defining the fixed splits.
        output_dir: Parent directory for the run-specific checkpoint folder.
        num_workers: Number of data loader workers used for video encoding.

    Returns:
        A tuple containing the best checkpoint path, loss history, and
        accuracy history. Both histories use the keys "train" and "val".
        Accuracy values in the histories are fractions between 0 and 1.
    """
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select a CUDA GPU with bfloat16 support.")

    device = torch.device("cuda")
    manifest_path = Path(manifest_path)
    encoder = FrozenVideoEncoder()
    loaders = build_window_loaders(
        encoder, manifest_path, Path(video_root), device, num_workers
    )
    torch.manual_seed(42)
    model = LRCN(100, 1, 0.1, 50, embedding_size=encoder.embedding_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    config = {
        "encoder": "R(2+1)D-18 Kinetics-400",
        "split_manifest": str(manifest_path),
        "seed": 42,
        "batch_size": 128,
        "epochs": 15,
        "learning_rate": 0.001,
        "temporal_pool": "last",
    }

    with wandb.init(
        project="assignment4-ucf50",
        name="repo-r2plus1d18-lrcn",
        job_type="training",
        config=config,
    ) as wandb_run:
        run_dir = Path(output_dir) / wandb_run.id
        run_dir.mkdir(parents=True, exist_ok=False)
        _, loss_history, accuracy_history = train(
            loaders, model, nn.CrossEntropyLoss(), optimizer, None, device,
            run_dir, n_epochs=15, wandb_run=wandb_run,
        )

    return run_dir / "best_model_wts.pt", loss_history, accuracy_history

def evaluate_window_lrcn(video_root, manifest_path, checkpoint_path, num_workers=2):
    """Evaluate a validation-selected checkpoint on the held-out UCF50 split.

    Encode the test clips with the same frozen backbone used for training.
    Log mean cross-entropy, top-1 accuracy, and clip counts to a separate
    Weights & Biases run.

    Args:
        video_root: Directory containing the UCF50 action-class folders.
        manifest_path: Path to the same split JSON used during training.
        checkpoint_path: Path returned by train_window_lrcn().
        num_workers: Number of data loader workers used for video encoding.

    Returns:
        Top-1 test accuracy as a fraction between 0 and 1.
    """
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select a CUDA GPU with bfloat16 support.")

    device = torch.device("cuda")
    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _, _, test_samples = load_manifest_splits(manifest_path, ".", 50)

    encoder = FrozenVideoEncoder()
    model = LRCN(100, 1, 0.1, 50, embedding_size=encoder.embedding_size)
    model.load_state_dict(state)
    model.to(device)

    dataset = VideoWindowDataset(test_samples, Path(video_root))
    test_data = encode_split(encoder, dataset, device, num_workers)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    with wandb.init(
        project="assignment4-ucf50",
        name="repo-r2plus1d18-lrcn-held-out-test",
        job_type="evaluation",
        config={
            "split_manifest": str(manifest_path),
            "checkpoint": str(checkpoint_path),
            "parent_run": checkpoint_path.parent.name,
        },
    ) as wandb_run:
        _, _, accuracy = test(model, test_loader, device, wandb_run=wandb_run)

    print(f"Held-out test accuracy: {100 * accuracy:.2f}%")
    return accuracy

def main(args):
    """
    Main function to execute training or evaluation based on the parsed command-line arguments.

    For training:
        - Loads the dataset from the specified frame directory.
        - Splits the dataset into training, validation, and test sets.
        - Saves the splits for later use.
        - Creates the training and validation DataLoaders.
        - Initializes the model (LRCN) with specified hyperparameters.
        - Defines the loss function, optimizer, and learning rate scheduler.
        - Runs the training loop and saves the best model weights.

    For evaluation:
        - Uses the test split from --split_manifest when supplied.
          Otherwise, loads the test split saved in splits.npy.
        - Creates the test DataLoader and loads the trained checkpoint.
        - Logs test loss and accuracy to a separate Weights & Biases run
          and prints the overall test accuracy.
          
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Dataset parameters
    frame_dir = args.frame_dir
    tr_size = args.train_size
    ts_size = args.test_size
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone
    ckpt = args.ckpt

    # Training parameters
    mode = args.mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    # Initialize the model (LRCN)
    model = LRCN(
        hidden_size=rnn_hidden_size,
        n_layers=rnn_n_layers,
        dropout_rate=dropout,
        n_classes=n_classes,
        pretrained=pretrained,
        cnn_model=cnn_backbone,
        temporal_pool=args.temporal_pool,
    )

    if mode == 'train':
        # Load dataset and split into train/validation/test
        if args.split_manifest is not None:
            tr_split, val_split, ts_split = load_manifest_splits(
                args.split_manifest, frame_dir, n_classes
            )
        else:
            vid_dataset, _ = load_dataset(frame_dir)
            tr_split, val_split, ts_split = dataset_split(
                vid_dataset, tr_size, ts_size
            )

        # Save the splits for reproducibility and later use in evaluation
        splits = {'train': np.array(tr_split),
                  'val': np.array(val_split),
                  'test': np.array(ts_split)}
        np.save('./splits.npy', splits)

        # Create PyTorch Datasets and DataLoaders for train and validation
        tr_dataset = VideoDataset(tr_split, fr_per_vid, tr_transforms)
        val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms)
        dataloaders = train_val_dloaders(tr_dataset, val_dataset, batch_size, model_type)

        # Define the loss function, optimizer, and learning rate scheduler
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        os.makedirs("./models", exist_ok=True)
        optim_model_dir = './models'

        # Main training procedure
        model.to(device)
        with wandb.init(
            project="assignment4-ucf50",
            name="lrcn-resnet34-baseline",
            config=vars(args),
        ) as wandb_run:
            model, loss_hist, acc_hist = train(
                dataloaders,
                model,
                loss_func,
                opt,
                lr_scheduler,
                device,
                optim_model_dir,
                n_epochs,
                wandb_run=wandb_run,
            )

    elif mode == "eval":
        if args.split_manifest is not None:
            _, _, ts_split = load_manifest_splits(
                args.split_manifest, frame_dir, n_classes
            )
        else:
            splits = np.load("./splits.npy", allow_pickle=True).item()
            ts_split = [(path, int(label)) for path, label in splits["test"]]

        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        dataloaders = test_dloaders(ts_dataset, batch_size, model_type)

        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.to(device)

        config = {
            **vars(args),
            "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        }
        with wandb.init(
            project="assignment4-ucf50",
            name=f"lrcn-{cnn_backbone}-held-out-test",
            job_type="evaluation",
            config=config,
        ) as wandb_run:
            _, _, accuracy = test(
                model, dataloaders["test"], device, wandb_run=wandb_run
            )

        print(f"The overall test accuracy is {100 * accuracy:.4f}%.")

if __name__ == "__main__":
    args = args_parser()
    main(args)
