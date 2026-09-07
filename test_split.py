"""Tests for the train/validation split and class weighting."""
import torch

from split import positive_weights, split_indices_by_video


def pairs(*counts):
    """Build sample_pairs for videos v0, v1, ... with the given frame counts."""
    out = []
    for video, count in enumerate(counts):
        for frame in range(count):
            out.append((f"v{video}.mp4", f"v{video}.json", frame))
    return out


def test_no_video_appears_on_both_sides():
    sample_pairs = pairs(10, 10, 10, 10, 10)
    train, val = split_indices_by_video(sample_pairs)

    train_videos = {sample_pairs[i][0] for i in train}
    val_videos = {sample_pairs[i][0] for i in val}

    assert train_videos and val_videos
    assert train_videos.isdisjoint(val_videos)


def test_every_sample_is_used_exactly_once():
    sample_pairs = pairs(7, 5, 9, 4)
    train, val = split_indices_by_video(sample_pairs)

    assert sorted(train + val) == list(range(len(sample_pairs)))


def test_the_split_is_deterministic():
    sample_pairs = pairs(6, 6, 6, 6, 6)

    assert split_indices_by_video(sample_pairs) == split_indices_by_video(sample_pairs)


def test_a_single_video_holds_out_its_tail():
    sample_pairs = pairs(10)
    train, val = split_indices_by_video(sample_pairs, val_fraction=0.2)

    assert train == [0, 1, 2, 3, 4, 5, 6, 7]
    assert val == [8, 9]


def test_empty_input():
    assert split_indices_by_video([]) == ([], [])


def test_a_rare_action_is_weighted_up():
    # Action 0 fires in 1 frame of 10; action 1 fires in 5.
    labels = torch.zeros(10, 2)
    labels[0, 0] = 1
    labels[:5, 1] = 1

    weights = positive_weights(labels)

    assert weights[0].item() == 9.0   # 9 negatives / 1 positive
    assert weights[1].item() == 1.0   # balanced
    assert weights[0] > weights[1]


def test_an_action_that_never_fires_gets_a_neutral_weight():
    labels = torch.zeros(8, 3)
    labels[:4, 1] = 1

    weights = positive_weights(labels)

    assert torch.isfinite(weights).all()
    assert weights[0].item() == 1.0
    assert weights[2].item() == 1.0


def test_weights_are_capped():
    # One positive in 1000 frames would otherwise weight 999.
    labels = torch.zeros(1000, 1)
    labels[0, 0] = 1

    assert positive_weights(labels, cap=20.0)[0].item() == 20.0
