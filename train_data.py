from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY = True


def _balanced_subset_indices(targets, per_class: int = 96, num_classes: int = 100):
    """Return a deterministic class-balanced subset.

    CIFAR100 has 500 train samples per class.  A balanced subset avoids batches
    dominated by a few classes when the evaluation budget is only 8192 samples.
    """
    buckets = [[] for _ in range(num_classes)]
    for idx, y in enumerate(targets):
        if len(buckets[y]) < per_class:
            buckets[y].append(idx)
        if all(len(b) >= per_class for b in buckets):
            break
    indices = []
    for k in range(per_class):
        for c in range(num_classes):
            indices.append(buckets[c][k])
    return indices


def get_train_dataset_loader(data_dir, batch_size, generator_train):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    base_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )

    # 9600 samples lets the official runner draw 8192 examples without cycling
    # for the recommended 32 x 256 setting, while keeping class balance.
    indices = _balanced_subset_indices(base_dataset.targets, per_class=96)
    train_dataset = Subset(base_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train,
        drop_last=True,
    )
    return train_dataset, train_loader
