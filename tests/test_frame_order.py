"""Tests for chronological video frame loading."""

from pathlib import Path

import torch
from PIL import Image

from video_datasets import VideoDataset


def test_frames_are_sorted_before_applying_limit(tmp_path, monkeypatch):
    """Select frames 1 and 2 when the input order is 10, 2, 1."""
    paths = [tmp_path / f"frame{number}.jpg" for number in (10, 2, 1)]
    for path, value in zip(paths, (100, 20, 10)):
        Image.new("L", (4, 4), color=value).save(path)

    monkeypatch.setattr(Path, "glob", lambda _path, _pattern: iter(paths))
    dataset = VideoDataset(
        [(str(tmp_path), 0)],
        fr_per_vid=2,
        transforms=lambda image: torch.tensor([[[image.getpixel((0, 0))]]]),
    )

    frames, label = dataset[0]

    assert frames[:, 0, 0, 0].tolist() == [10, 20]
    assert label == 0