"""Tests for the UCF50 train, validation, and test split."""

from collections import Counter
from itertools import combinations
from pathlib import Path

from video_datasets import dataset_split


def test_dataset_split_keeps_source_groups_together():
    """Keep each source group in one split and include every clip once."""
    clips = {
        f"{action}/v_{action}_g{group:02d}_c{clip:02d}": label
        for label, action in enumerate(("Biking", "Diving"))
        for group in range(1, 26)
        for clip in range(1, 4)
    }

    splits = dataset_split(clips, tr_ratio=0.75, ts_ratio=0.15, seed=42)
    assert Counter(sample for split in splits for sample in split) == Counter(
        clips.items()
    )

    groups_by_split = [
        {(label, Path(path).name.rsplit("_c", 1)[0]) for path, label in split}
        for split in splits
    ]
    for first, second in combinations(groups_by_split, 2):
        assert first.isdisjoint(second)

    for groups, expected_count in zip(groups_by_split, (19, 2, 4)):
        assert Counter(label for label, _ in groups) == {
            0: expected_count,
            1: expected_count,
        }