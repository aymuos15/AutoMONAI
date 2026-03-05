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
    img_size,
    spatial_dims=2,
    augment=False,
    aug_prob=0.5,
    norm=None,
    crop=None,
    is_train=True,
    extra_transforms=None,
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

    # Apply extra transforms from featurelist
    if extra_transforms and is_train:
        pipeline.extend(_build_extra_transforms(extra_transforms, aug_prob, spatial_dims, size))

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


def _build_extra_transforms(transform_names, prob, spatial_dims, size):
    """Build a list of extra MONAI transform objects from their short names."""
    from monai.transforms import (
        RandAffine,
        RandRotate90,
        CropForeground,
        SpatialPad,
        BorderPad,
        DivisiblePad,
        GridPatch,
        RandGibbsNoise,
        RandKSpaceSpikeNoise,
        RandBiasField,
        RandCoarseDropout,
        RandCoarseShuffle,
        RandHistogramShift,
        RandShiftIntensity,
        RandScaleIntensity,
        RandGaussianSmooth,
        RandGaussianSharpen,
        MaskIntensity,
        ScaleIntensityRange,
        ThresholdIntensity,
        ClipIntensityPercentiles,
    )

    transforms = []
    for name in transform_names:
        if name == "rand_affine":
            transforms.append(
                RandAffine(prob=prob, rotate_range=(0.3,) * spatial_dims, spatial_size=size)
            )
        elif name == "rand_elastic_2d":
            from monai.transforms import Rand2DElastic

            if spatial_dims == 2:
                transforms.append(
                    Rand2DElastic(
                        prob=prob,
                        spacing=(20, 20),
                        magnitude_range=(1, 2),
                        spatial_size=size,
                    )
                )
        elif name == "rand_elastic_3d":
            from monai.transforms import Rand3DElastic

            if spatial_dims == 3:
                transforms.append(
                    Rand3DElastic(
                        prob=prob,
                        sigma_range=(5, 8),
                        magnitude_range=(100, 200),
                        spatial_size=size,
                    )
                )
        elif name == "rand_crop_pos_neg":
            from monai.transforms import RandCropByPosNegLabel

            transforms.append(RandCropByPosNegLabel(spatial_size=size, pos=1, neg=1, num_samples=1))
        elif name == "crop_foreground":
            transforms.append(CropForeground())
        elif name == "rand_rotate90":
            transforms.append(RandRotate90(prob=prob, spatial_axes=(0, 1)))
        elif name == "rand_spatial_crop_samples":
            from monai.transforms import RandSpatialCropSamples

            transforms.append(RandSpatialCropSamples(roi_size=size, num_samples=2))
        elif name == "spatial_pad":
            transforms.append(SpatialPad(spatial_size=size))
        elif name == "border_pad":
            transforms.append(BorderPad(spatial_border=4))
        elif name == "divisible_pad":
            transforms.append(DivisiblePad(k=16))
        elif name == "grid_patch":
            transforms.append(GridPatch(patch_size=size))
        elif name == "rand_gibbs_noise":
            transforms.append(RandGibbsNoise(prob=prob))
        elif name == "rand_kspace_spike_noise":
            transforms.append(RandKSpaceSpikeNoise(prob=prob))
        elif name == "rand_bias_field":
            transforms.append(RandBiasField(prob=prob))
        elif name == "rand_coarse_dropout":
            holes = tuple(max(1, s // 8) for s in size)
            transforms.append(RandCoarseDropout(prob=prob, holes=1, spatial_size=holes))
        elif name == "rand_coarse_shuffle":
            holes = tuple(max(1, s // 8) for s in size)
            transforms.append(RandCoarseShuffle(prob=prob, holes=1, spatial_size=holes))
        elif name == "rand_histogram_shift":
            transforms.append(RandHistogramShift(prob=prob))
        elif name == "rand_shift_intensity":
            transforms.append(RandShiftIntensity(prob=prob, offsets=0.1))
        elif name == "rand_scale_intensity":
            transforms.append(RandScaleIntensity(prob=prob, factors=0.1))
        elif name == "rand_gaussian_smooth":
            transforms.append(RandGaussianSmooth(prob=prob))
        elif name == "rand_gaussian_sharpen":
            transforms.append(RandGaussianSharpen(prob=prob))
        elif name == "mask_intensity":
            transforms.append(MaskIntensity(mask_data=None))
        elif name == "clip_intensity_percentiles":
            transforms.append(ClipIntensityPercentiles(lower=5, upper=95))
        elif name == "scale_intensity_range":
            transforms.append(
                ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True)
            )
        elif name == "threshold_intensity":
            transforms.append(ThresholdIntensity(threshold=0.5, above=True, cval=0.0))
    return transforms
