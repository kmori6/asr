import torch

from asr.modules.frontend import SpecAugment


def test_spec_augment_masks_only_valid_features_during_training() -> None:
    torch.manual_seed(0)
    features = torch.ones(2, 8, 6)
    features[1, 5:] = -1.0
    original_features = features.clone()
    feature_lengths = torch.tensor([8, 5])
    spec_augment = SpecAugment(
        num_frequency_masks=2,
        max_frequency_mask_width=3,
        num_time_masks=2,
        max_time_mask_width=4,
    )

    augmented = spec_augment(features, feature_lengths)

    assert augmented.shape == features.shape
    assert torch.any(augmented[:, :5] == 0.0)
    torch.testing.assert_close(augmented[1, 5:], features[1, 5:])
    torch.testing.assert_close(features, original_features)


def test_spec_augment_is_disabled_during_evaluation() -> None:
    features = torch.randn(2, 8, 6)
    feature_lengths = torch.tensor([8, 5])
    spec_augment = SpecAugment(
        num_frequency_masks=2,
        max_frequency_mask_width=3,
        num_time_masks=2,
        max_time_mask_width=4,
    ).eval()

    augmented = spec_augment(features, feature_lengths)

    torch.testing.assert_close(augmented, features)
