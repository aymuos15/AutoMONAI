def get_model(model_name, in_channels, out_channels, spatial_dims=2, img_size=128):
    from monai.networks.nets import UNet, AttentionUnet, SegResNet, SwinUNETR

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
    else:
        raise ValueError(f"Unknown model: {model_name}")
