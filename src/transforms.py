import numpy as np
from PIL import Image


class PILLoadImage:
    def __call__(self, path, spatial_dims=2):
        if path.endswith(".nii.gz"):
            from nibabel.loadsave import load

            img = load(path)
            data = img.get_fdata()  # type: ignore[attr-defined]
            data = data.astype(np.float32)

            if spatial_dims == 2:
                if data.ndim == 3:
                    data = data[:, :, data.shape[2] // 2]
                data = np.expand_dims(data, axis=0)
            else:
                if data.ndim == 3:
                    data = np.expand_dims(data, axis=0)

            return data
        else:
            img = Image.open(path)
            img = img.convert("L")
            arr = np.array(img, dtype=np.float32)
            arr = np.expand_dims(arr, axis=0)
            return arr


def get_transforms(
    img_size, spatial_dims=2, augment=False, aug_prob=0.5, norm=None, crop=None, is_train=True
):
    from monai.transforms import (
        ScaleIntensity,
        NormalizeIntensity,
        Resize,
        ToTensor,
        Compose,
        RandRotate,
        RandFlip,
        RandZoom,
        RandGaussianNoise,
        RandAdjustContrast,
        CenterSpatialCrop,
        RandSpatialCrop,
    )

    if spatial_dims == 3:
        if isinstance(img_size, tuple):
            size = img_size
        else:
            size = (img_size, img_size, img_size)
    else:
        if isinstance(img_size, tuple):
            size = img_size
        else:
            size = (img_size, img_size)

    pipeline = [PILLoadImage()]

    if norm is None:
        norm = []
    if crop is None:
        crop = []

    if "minmax" in norm:
        pipeline.append(ScaleIntensity())
    if "zscore" in norm:
        pipeline.append(NormalizeIntensity())

    has_center = "center" in crop
    has_random = "random" in crop

    if has_random and is_train:
        pipeline.append(RandSpatialCrop(size, random_size=False))
    elif has_center:
        pipeline.append(CenterSpatialCrop(size))

    pipeline.append(Resize(size))

    if augment:
        pipeline.extend(
            [
                RandRotate(prob=aug_prob, range_x=0.3, range_y=0.3, keep_size=True),
                RandFlip(prob=aug_prob, spatial_axis=None),
                RandZoom(prob=aug_prob, min_zoom=0.8, max_zoom=1.2, keep_size=True),
                RandGaussianNoise(prob=aug_prob, mean=0.0, std=0.1),
                RandAdjustContrast(prob=aug_prob, gamma=(0.5, 1.5)),
            ]
        )

    pipeline.append(ToTensor())

    return Compose(pipeline)
