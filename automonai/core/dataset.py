from pathlib import Path

from monai.data import (
    Dataset,
    CacheDataset,
    PersistentDataset,
    SmartCacheDataset,
)

from .config import DATASET_ROOT


def list_datasets():
    from .config import DATASETS

    return list(DATASETS.keys())


def get_train_files(dataset_name):
    dataset_path = Path(DATASET_ROOT) / dataset_name
    images_dir = dataset_path / "imagesTr"
    labels_dir = dataset_path / "labelsTr"

    image_files = (
        sorted(images_dir.glob("*_0000.png"))
        + sorted(images_dir.glob("*_0000.nii.gz"))
        + sorted(images_dir.glob("*_0000.tif"))
    )

    is_3d = any(f.suffixes == [".nii", ".gz"] or f.suffix == ".nii.gz" for f in image_files)

    files = []
    for img in image_files:
        base_name = img.name.split("_")[0]
        if img.name.startswith("hippocampus"):
            base_name = "_".join(img.name.split("_")[:2])

        if img.name.endswith(".nii.gz"):
            label_name = base_name + ".nii.gz"
        elif img.name.endswith(".tif"):
            label_name = base_name + ".png"
        else:
            label_name = base_name + ".png"

        label_path = labels_dir / label_name
        if label_path.exists():
            files.append({"image": str(img), "label": str(label_path)})

    return files, is_3d


def get_test_files(dataset_name):
    dataset_path = Path(DATASET_ROOT) / dataset_name
    images_dir = dataset_path / "imagesTs"

    image_files = (
        sorted(images_dir.glob("*_0000.png"))
        + sorted(images_dir.glob("*_0000.nii.gz"))
        + sorted(images_dir.glob("*_0000.tif"))
    )

    is_3d = any(f.suffixes == [".nii", ".gz"] or f.suffix == ".nii.gz" for f in image_files)

    return [{"image": str(f)} for f in image_files], is_3d


def get_test_files_with_labels(dataset_name):
    dataset_path = Path(DATASET_ROOT) / dataset_name
    images_dir = dataset_path / "imagesTs"
    labels_dir = dataset_path / "labelsTs"

    image_files = (
        sorted(images_dir.glob("*_0000.png"))
        + sorted(images_dir.glob("*_0000.nii.gz"))
        + sorted(images_dir.glob("*_0000.tif"))
    )

    is_3d = any(f.suffixes == [".nii", ".gz"] or f.suffix == ".nii.gz" for f in image_files)

    files = []
    for img in image_files:
        base_name = img.name.split("_")[0]
        if img.name.startswith("hippocampus"):
            base_name = "_".join(img.name.split("_")[:2])

        if img.name.endswith(".nii.gz"):
            label_name = base_name + ".nii.gz"
        elif img.name.endswith(".tif"):
            label_name = base_name + ".png"
        else:
            label_name = base_name + ".png"

        label_path = labels_dir / label_name
        if label_path.exists():
            files.append({"image": str(img), "label": str(label_path)})

    return files, is_3d


def split_train_val(files, mode="none", val_ratio=0.2, n_folds=5, fold=0, seed=42, split_file=None):
    """Split training files into train/val sets. Returns (train_files, val_files)."""
    import random

    if mode == "none":
        return files, []

    if mode == "holdout":
        shuffled = list(files)
        random.Random(seed).shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - val_ratio))
        return shuffled[:split_idx], shuffled[split_idx:]

    if mode == "kfold":
        shuffled = list(files)
        random.Random(seed).shuffle(shuffled)
        fold_size = len(shuffled) // n_folds
        remainder = len(shuffled) % n_folds
        # Build fold boundaries
        folds = []
        start = 0
        for i in range(n_folds):
            end = start + fold_size + (1 if i < remainder else 0)
            folds.append(shuffled[start:end])
            start = end
        val_files = folds[fold]
        train_files = [f for i, chunk in enumerate(folds) if i != fold for f in chunk]
        return train_files, val_files

    if mode == "custom":
        import json as _json
        with open(split_file) as f:
            split_data = _json.load(f)
        train_names = set(split_data["train"])
        val_names = set(split_data["val"])
        train_files = [f for f in files if Path(f["image"]).name in train_names]
        val_files = [f for f in files if Path(f["image"]).name in val_names]
        return train_files, val_files

    return files, []


class TrainDataset(Dataset):
    def __init__(self, files, transform=None, label_transform=None):
        super().__init__(files, transform)
        self.label_transform = label_transform

    def __getitem__(self, index):
        data = self.data[index]
        image = self.transform(data["image"])
        label = self.label_transform(data["label"]) if self.label_transform else None
        return {"image": image, "label": label}


class TestDataset(Dataset):
    def __init__(self, files, transform=None):
        super().__init__(files, transform)

    def __getitem__(self, index):
        data = self.data[index]
        image = self.transform(data["image"])
        return {"image": image, "filename": data["image"]}


class DictTransform:
    def __init__(self, image_transform, label_transform):
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __call__(self, data):
        return {
            "image": self.image_transform(data["image"]),
            "label": self.label_transform(data["label"]),
        }


def create_train_dataset(
    dataset_class_name,
    files,
    transform,
    label_transform,
    cache_rate=1.0,
    replace_rate=None,
    cache_dir=None,
):
    if dataset_class_name == "Dataset":
        return TrainDataset(files, transform=transform, label_transform=label_transform)
    elif dataset_class_name == "CacheDataset":
        return CacheDataset(files, transform=transform, cache_rate=cache_rate)
    elif dataset_class_name == "PersistentDataset":
        return PersistentDataset(files, transform=transform, cache_dir=cache_dir)
    elif dataset_class_name == "SmartCacheDataset":
        ds = SmartCacheDataset(
            files, transform=transform, cache_rate=cache_rate, replace_rate=replace_rate or 0.1
        )
        ds.start()
        return ds
    elif dataset_class_name == "LMDBDataset":
        from monai.data import LMDBDataset

        lmdb_path = cache_dir or "/tmp/automonai_lmdb"
        return LMDBDataset(
            files, transform=transform, lmdb_kwargs={"map_size": 1 << 30}, cache_dir=lmdb_path
        )
    elif dataset_class_name == "CacheNTransDataset":
        from monai.data import CacheNTransDataset

        return CacheNTransDataset(files, transform=transform, cache_n_trans=3, cache_dir=cache_dir)
    elif dataset_class_name == "ArrayDataset":
        from monai.data import ArrayDataset

        images = [f["image"] for f in files]
        labels = [f["label"] for f in files]
        return ArrayDataset(
            img=images, img_transform=transform, seg=labels, seg_transform=label_transform
        )
    elif dataset_class_name == "ZipDataset":
        from monai.data import ZipDataset

        img_ds = Dataset([f["image"] for f in files], transform=transform)
        lbl_ds = Dataset([f["label"] for f in files], transform=label_transform)
        return ZipDataset([img_ds, lbl_ds])
    elif dataset_class_name == "GridPatchDataset":
        from monai.data import GridPatchDataset

        return GridPatchDataset(data=files, transform=transform, patch_size=(64, 64))
    elif dataset_class_name == "PatchDataset":
        from monai.data import PatchDataset

        ds = Dataset(files, transform=transform)
        return PatchDataset(data=ds, patch_func=lambda x: [x], samples_per_image=1)
    elif dataset_class_name == "DecathlonDataset":
        from monai.apps import DecathlonDataset as MonaiDecathlonDataset

        return MonaiDecathlonDataset(
            root_dir=str(cache_dir or "/tmp/automonai_decathlon"),
            task="Task01_BrainTumour",
            section="training",
            transform=transform,
            download=False,
        )
    else:
        raise ValueError(f"Unknown dataset class: {dataset_class_name}")


def create_inference_dataset(dataset_class_name, files, transform, cache_rate=1.0, cache_dir=None):
    if dataset_class_name == "Dataset":
        return TestDataset(files, transform=transform)
    elif dataset_class_name == "CacheDataset":
        return CacheDataset(files, transform=transform, cache_rate=cache_rate)
    elif dataset_class_name == "PersistentDataset":
        return PersistentDataset(files, transform=transform, cache_dir=cache_dir)
    elif dataset_class_name == "SmartCacheDataset":
        return SmartCacheDataset(files, transform=transform, cache_rate=cache_rate)
    elif dataset_class_name == "LMDBDataset":
        from monai.data import LMDBDataset

        lmdb_path = cache_dir or "/tmp/automonai_lmdb_infer"
        return LMDBDataset(
            files, transform=transform, lmdb_kwargs={"map_size": 1 << 30}, cache_dir=lmdb_path
        )
    elif dataset_class_name == "CacheNTransDataset":
        from monai.data import CacheNTransDataset

        return CacheNTransDataset(files, transform=transform, cache_n_trans=3, cache_dir=cache_dir)
    elif dataset_class_name == "ArrayDataset":
        from monai.data import ArrayDataset

        images = [f["image"] for f in files]
        return ArrayDataset(img=images, img_transform=transform)
    elif dataset_class_name in (
        "ZipDataset",
        "GridPatchDataset",
        "PatchDataset",
        "DecathlonDataset",
    ):
        # Fall back to basic dataset for inference with these types
        return TestDataset(files, transform=transform)
    else:
        raise ValueError(f"Unknown dataset class: {dataset_class_name}")
