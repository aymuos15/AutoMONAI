def get_model(
    model_name, in_channels, out_channels, spatial_dims=2, img_size=128, deep_supervision=False
):
    from monai.networks.nets import (
        UNet,
        AttentionUnet,
        SegResNet,
        SwinUNETR,
        BasicUNet,
        BasicUNetPlusPlus,
        DynUNet,
        VNet,
        HighResNet,
        UNETR,
        SegResNetVAE,
        FlexibleUNet,
    )

    if isinstance(img_size, int):
        roi_size = (img_size,) * spatial_dims
    else:
        roi_size = tuple(img_size)

    if model_name == "unet":
        return UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif model_name == "attention_unet":
        return AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
    elif model_name == "segresnet":
        return SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
    elif model_name == "swinunetr":
        return SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=24,
            spatial_dims=spatial_dims,
        )
    elif model_name == "basicunet":
        return BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif model_name == "basicunetplusplus":
        return BasicUNetPlusPlus(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif model_name == "dynunet":
        # DynUNet requires kernel_size and strides as lists
        kernel_size = [[3] * spatial_dims] * 5
        strides = [[1] * spatial_dims] + [[2] * spatial_dims] * 4
        return DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=strides[1:],
            deep_supervision=deep_supervision,
            deep_supr_num=3 if deep_supervision else 1,
        )
    elif model_name == "vnet":
        return VNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif model_name == "highresnet":
        return HighResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif model_name == "unetr":
        return UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=roi_size,
            spatial_dims=spatial_dims,
        )
    elif model_name == "segresnetvae":
        return SegResNetVAE(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            input_image_size=roi_size,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
    elif model_name == "segresnetds":
        from monai.networks.nets import SegResNetDS

        return SegResNetDS(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
    elif model_name == "segresnetds2":
        from monai.networks.nets import SegResNetDS2

        return SegResNetDS2(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
    elif model_name == "flexibleunet":
        return FlexibleUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            backbone="efficientnet-b0",
            spatial_dims=spatial_dims,
        )
    elif model_name == "dints":
        from monai.networks.nets import DiNTS, TopologySearch

        topo = TopologySearch(
            channel_mul=0.5,
            num_blocks=6,
            num_depths=3,
            use_downsample=True,
            spatial_dims=spatial_dims,
        )
        return DiNTS(
            dints_space=topo,
            in_channels=in_channels,
            num_classes=out_channels,
            use_downsample=True,
            node_a=torch.randn(6, 8),
        )
    elif model_name.startswith("mednext"):
        from monai.networks.nets import MedNeXt

        size_configs = {
            "mednext_s": {
                "blocks_down": (2, 2, 2, 2),
                "blocks_bottleneck": 2,
                "blocks_up": (2, 2, 2, 2),
            },
            "mednext_m": {
                "blocks_down": (2, 2, 2, 2),
                "blocks_bottleneck": 2,
                "blocks_up": (2, 2, 2, 2),
                "encoder_expansion_ratio": 3,
                "decoder_expansion_ratio": 3,
            },
            "mednext_b": {
                "blocks_down": (2, 4, 8, 8),
                "blocks_bottleneck": 8,
                "blocks_up": (2, 4, 8, 8),
            },
            "mednext_l": {
                "blocks_down": (2, 4, 8, 8),
                "blocks_bottleneck": 8,
                "blocks_up": (2, 4, 8, 8),
                "encoder_expansion_ratio": 3,
                "decoder_expansion_ratio": 3,
            },
        }
        size_cfg = size_configs.get(model_name, size_configs["mednext_s"])
        # MedNeXt kernel_size must be odd
        return MedNeXt(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            deep_supervision=deep_supervision,
            **size_cfg,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Needed for DiNTS random arch sampling
import torch  # noqa: E402
