"""Train/validation splitting and class statistics for the recorded gameplay.

Kept apart from train.py so both can be tested without loading video.
"""
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch


def split_indices_by_video(
    sample_pairs: Sequence[Tuple[str, str, int]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Split sample indices into train and validation, keeping every frame of a
    video on the same side.

    Consecutive frames of one recording are nearly identical, so a split that
    scatters them across both sides puts near-copies of training frames into
    validation and reports a score that reflects memorisation. Splitting whole
    videos is the honest version.

    Falls back to a contiguous split when there are too few videos to divide,
    which at least keeps the validation frames away from their neighbours.
    """
    by_video: Dict[str, List[int]] = defaultdict(list)
    for index, (video_path, _json_path, _frame_idx) in enumerate(sample_pairs):
        by_video[video_path].append(index)

    videos = sorted(by_video)
    if not videos:
        return [], []

    n_val = int(round(len(videos) * val_fraction))

    if n_val == 0 or n_val == len(videos):
        # One usable video: hold out the tail of each recording instead.
        train, val = [], []
        for video in videos:
            indices = by_video[video]
            cut = int(len(indices) * (1 - val_fraction))
            train.extend(indices[:cut])
            val.extend(indices[cut:])
        return train, val

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(videos), generator=generator).tolist()
    val_videos = {videos[i] for i in order[:n_val]}

    train = [i for v in videos if v not in val_videos for i in by_video[v]]
    val = [i for v in videos if v in val_videos for i in by_video[v]]
    return train, val


def positive_weights(labels: torch.Tensor, cap: float = 20.0) -> torch.Tensor:
    """
    Per-action `pos_weight` for BCEWithLogitsLoss: negatives divided by
    positives, so a rarely pressed control is not drowned out by the frames
    where nothing happens.

    Without this, predicting "no action" everywhere is a low-loss solution and
    the model has little reason to leave it. The cap keeps a control that is
    almost never active from dominating the gradient, and an action that never
    occurs gets weight 1 rather than an infinity.
    """
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives

    weights = torch.ones_like(positives)
    seen = positives > 0
    weights[seen] = (negatives[seen] / positives[seen]).clamp(max=cap)
    return weights
