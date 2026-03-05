"""Inferer factory for different inference strategies."""


def get_inferer(inferer_name, model=None, roi_size=None, spatial_dims=2):
    """Create an inferer based on the given name.

    Args:
        inferer_name: One of 'simple', 'sliding_window', 'patch', 'saliency', 'slice'
        model: The model (needed for saliency inferer)
        roi_size: Tuple of spatial dimensions for sliding window / patch
        spatial_dims: 2 or 3

    Returns:
        A MONAI inferer instance, or None for simple (direct model call).
    """
    if inferer_name == "simple" or inferer_name is None:
        return None

    if roi_size is None:
        roi_size = (128,) * spatial_dims

    if inferer_name == "sliding_window":
        from monai.inferers import SlidingWindowInferer

        return SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=1,
            overlap=0.25,
            mode="gaussian",
        )
    elif inferer_name == "patch":
        from monai.inferers import PatchInferer

        return PatchInferer(
            splitter="SlidingWindowSplitter",
            merger_cls="AvgMerger",
            patch_size=roi_size,
        )
    elif inferer_name == "saliency":
        from monai.inferers import SaliencyInferer

        return SaliencyInferer(cam_name="GradCAM", target_layers="model.layer4")
    elif inferer_name == "slice":
        from monai.inferers import SliceInferer

        return SliceInferer(
            roi_size=roi_size[:2] if spatial_dims == 3 else roi_size,
            spatial_dim=2 if spatial_dims == 3 else 0,
        )
    else:
        raise ValueError(f"Unknown inferer: {inferer_name}")


def run_inferer(inferer, model, inputs):
    """Run inference using the given inferer or direct model call."""
    if inferer is None:
        return model(inputs)
    return inferer(inputs, model)
