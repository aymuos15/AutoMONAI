import json
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


def load_dataset_info(dataset_name):
    json_path = Path(DATASET_ROOT) / dataset_name / "dataset.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    from .config import DATASETS

    return DATASETS.get(dataset_name)


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
