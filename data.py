from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def make_size_filter(min_size: int):
    """
    Build an ImageFolder `is_valid_file` callback that rejects images whose
    shorter edge is below `min_size`, so we never train on upscaled (blurry) data.

    Image.open reads only the header, so `.size` is cheap and does not decode pixels.
    """
    def is_valid_file(path: str) -> bool:
        if not path.lower().endswith(_IMG_EXTENSIONS):
            return False
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception:
            return False  # skip unreadable / corrupt files
        return min(width, height) >= min_size

    return is_valid_file


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    """Resize/crop to a square, optionally flip, and normalize images to [-1, 1]."""
    ops = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(ops)


def prepare_dataset(
    dataset: str,
    data_root: str,
    imagenet_path: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    min_image_size: int = 0,
) -> tuple[DataLoader, int]:
    """
    Build a training DataLoader for either CIFAR10 or a local ImageNet folder.

    The ImageNet folder is expected in torchvision ImageFolder layout:
        imagenet_path/<class_name>/<image>.jpeg
    Every sub-directory is treated as one class.

    :param dataset: Either "cifar10" or "imagenet".
    :param data_root: Where torchvision downloads/caches CIFAR10.
    :param imagenet_path: Root directory of the local ImageNet class folders.
    :param image_size: Square resolution to resize/crop images to.
    :param batch_size: Training batch size.
    :param num_workers: DataLoader worker processes.
    :param min_image_size: Drop images whose shorter edge is below this (0 disables).
        Only applies to the ImageNet folder; CIFAR10 is a fixed 32x32.
    :return: (train_loader, num_classes).
    """
    transform = build_transform(image_size, train=True)

    if dataset == "cifar10":
        train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        num_classes = 10
    elif dataset == "imagenet":
        is_valid_file = make_size_filter(min_image_size) if min_image_size > 0 else None
        train_set = datasets.ImageFolder(root=imagenet_path, transform=transform, is_valid_file=is_valid_file)
        num_classes = len(train_set.classes)
        print(f"ImageNet: kept {len(train_set)} images across {num_classes} classes"
              + (f" (filtered to shorter edge >= {min_image_size}px)" if min_image_size > 0 else ""))
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Expected 'cifar10' or 'imagenet'.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, num_classes
