"""Decode UCF50 videos and sample input windows for the video encoder."""

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from tqdm.auto import tqdm

WINDOWS_PER_VIDEO = 4
FRAMES_PER_WINDOW = 16
TARGET_FPS = 15


def decode_video(path):
    """Return resized RGB frames and the source frame rate."""
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.resize(frame, (171, 128), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames), fps if np.isfinite(fps) and fps > 0 else 30.0


def sample_windows(frames, fps):
    """Return ordered windows with dimensions (windows, time, channels, height, width)."""
    offsets = np.rint(np.arange(FRAMES_PER_WINDOW) * fps / TARGET_FPS).astype(int)
    last_start = max(0, len(frames) - 1 - offsets[-1])
    starts = np.rint(np.linspace(0, last_start, WINDOWS_PER_VIDEO)).astype(int)
    indices = np.minimum(starts[:, None] + offsets, len(frames) - 1)
    return torch.from_numpy(frames[indices]).permute(0, 1, 4, 2, 3)

class VideoWindowDataset(Dataset):
    """Load ordered windows for clips listed in the split manifest."""

    def __init__(self, samples, video_root):
        self.samples = samples
        self.video_root = video_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        clip, label = self.samples[index]
        frames, fps = decode_video(self.video_root / f"{clip}.avi")
        return sample_windows(frames, fps), label


class FrozenVideoEncoder(nn.Module):
    """Encode each video window with a frozen Kinetics-400 backbone."""

    def __init__(self):
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.preprocess = weights.transforms()
        self.backbone = r2plus1d_18(weights=weights)
        self.embedding_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad_(False)
        self.eval()

    def forward(self, windows):
        """Encode ordered RGB windows into one feature vector per window.

        Input shape is (batch, windows, frames, channels, height, width).
        Output shape is (batch, windows, 512). The UCF50 pipeline supplies
        four windows of 16 frames per clip and preserves their order.
        """
        inputs = self.preprocess(windows.flatten(0, 1))
        embeddings = self.backbone(inputs)
        return embeddings.reshape(
            windows.size(0), windows.size(1), self.embedding_size
        )


@torch.no_grad()
def encode_split(encoder, dataset, device, num_workers=2):
    """Encode one split with the frozen backbone in evaluation mode.

    Args:
        encoder: Frozen R(2+1)D-18 window encoder.
        dataset: VideoWindowDataset in manifest order.
        device: PyTorch device used for feature extraction.
        num_workers: Number of workers used to load and decode clips.

    Returns:
        A TensorDataset containing float32 CPU embeddings of shape
        (clips, windows, 512) and one class label per clip. Both tensors
        follow the dataset order.
    """
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=num_workers
    )
    encoder.to(device).eval()
    features, labels = [], []

    for windows, targets in tqdm(loader, desc="Encoding video windows"):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            embeddings = encoder(windows.to(device))
        features.append(embeddings.float().cpu())
        labels.append(targets)

    return TensorDataset(torch.cat(features), torch.cat(labels))