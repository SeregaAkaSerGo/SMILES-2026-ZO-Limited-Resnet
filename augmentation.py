"""Training and validation transforms for CIFAR100."""
import torchvision.transforms as T

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    if train:
        # Keep augmentations moderate: with zero-order optimization, excessive
        # stochastic augmentation increases the variance of scalar loss queries.
        return T.Compose(
            [
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                T.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
            ]
        )
    return T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ]
    )
