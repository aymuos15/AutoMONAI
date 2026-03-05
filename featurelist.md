# AutoMONAI Feature Tracker

Features from MONAI not yet supported. Check off as implemented.

---

## Models / Architectures

### Segmentation
- [ ] `BasicUNet` — Simplified U-Net
- [ ] `BasicUNetPlusPlus` — UNet++ with nested skip connections
- [ ] `DynUNet` — Dynamic U-Net (nnU-Net style)
- [ ] `VNet` — Volumetric segmentation
- [ ] `HighResNet` — Brain parcellation
- [ ] `UNETR` — UNet with Vision Transformers
- [ ] `SegResNetVAE` — SegResNet with variational autoencoder
- [ ] `SegResNetDS` — SegResNet with deep supervision
- [ ] `SegResNetDS2` — SegResNet DS variant 2
- [ ] `FlexibleUNet` — Configurable backbone U-Net
- [ ] `DiNTS` — Differentiable Neural Architecture Search
- [ ] `MedNeXt` (S/M/B/L) — ConvNeXt-style medical nets


---

## Loss Functions

### Segmentation Losses
- [ ] `DiceCELoss` — Dice + Cross Entropy combined
- [ ] `DiceFocalLoss` — Dice + Focal combined
- [ ] `GeneralizedDiceLoss` — Weighted Dice for class imbalance
- [ ] `GeneralizedWassersteinDiceLoss` — Wasserstein distance-based
- [ ] `GeneralizedDiceFocalLoss` — Generalized Dice + Focal
- [ ] `TverskyLoss` — Alpha/beta weighted Dice variant
- [ ] `HausdorffDTLoss` — Hausdorff Distance Transform loss
- [ ] `LogHausdorffDTLoss` — Log Hausdorff DT loss
- [ ] `SoftclDiceLoss` — Soft centerline Dice (tubular structures)
- [ ] `SoftDiceclDiceLoss` — Soft Dice + clDice combined
- [ ] `MaskedDiceLoss` — Spatially masked Dice
- [ ] `NACLLoss` — Neighbor-Aware Constrained Loss
- [ ] `AsymmetricUnifiedFocalLoss` — Asymmetric focal
- [ ] `SSIMLoss` — Structural similarity
- [ ] `PerceptualLoss` — LPIPS-based
- [ ] `DeepSupervisionLoss` — Multi-scale weighted loss


### Loss Wrappers
- [ ] `MaskedLoss` — Spatial mask wrapper
- [ ] `MultiScaleLoss` — Multi-scale smoothing wrapper

---

## Transforms

### Spatial
- [ ] `Spacing` / `Spacingd` — Resample to target voxel spacing
- [ ] `Orientation` / `Orientationd` — Canonical axis reorientation
- [ ] `RandAffine` — Random affine (combined rotate/scale/translate)
- [ ] `RandElasticDeformation` / `Rand2DElastic` / `Rand3DElastic` — Elastic deformation
- [ ] `RandCropByPosNegLabel` — Balanced foreground/background patch sampling
- [ ] `CropForeground` — Auto-crop to foreground
- [ ] `RandRotate90` — Random 90° rotation
- [ ] `RandSpatialCropSamples` — Multiple random crop samples
- [ ] `SpatialPad` / `BorderPad` / `DivisiblePad` — Padding transforms
- [ ] `GridPatch` / `GridSplit` — Grid-based patching

### Intensity
- [ ] `RandGibbsNoise` — Gibbs artifact augmentation
- [ ] `RandKSpaceSpikeNoise` — K-space spike noise
- [ ] `RandBiasField` — Bias field augmentation
- [ ] `RandCoarseDropout` — Cutout augmentation
- [ ] `RandCoarseShuffle` — Coarse shuffle
- [ ] `RandHistogramShift` — Histogram shifting
- [ ] `RandStdShiftIntensity` / `RandShiftIntensity` — Intensity shifting
- [ ] `RandScaleIntensity` — Random intensity scaling
- [ ] `RandGaussianSmooth` — Gaussian smoothing
- [ ] `RandGaussianSharpen` — Gaussian sharpening
- [ ] `GaussianSmooth` / `GaussianSharpen` — Deterministic smoothing/sharpening
- [ ] `MaskIntensity` — Mask-based intensity filtering
- [ ] `ClipIntensityPercentiles` — Percentile-based clipping
- [ ] `ScaleIntensityRange` — Range-based intensity scaling
- [ ] `ThresholdIntensity` — Intensity thresholding

---

## Metrics

- [ ] `HausdorffDistanceMetric` — Boundary distance
- [ ] `SurfaceDistanceMetric` — Average surface distance
- [ ] `SurfaceDiceMetric` — Normalized surface Dice (NSD)
- [ ] `GeneralizedDiceScore` — Weighted Dice score
- [ ] `ConfusionMatrixMetric` — Sensitivity, specificity, precision, F1
- [ ] `ROCAUCMetric` — AUC-ROC
- [ ] `FBetaScore` — F-beta score
- [ ] `PanopticQualityMetric` — PQ, SQ, RQ
- [ ] `CalibrationMetric` — Expected calibration error


---

## Inferers

- [ ] `SlidingWindowInferer` — Sliding window with overlap/blending
- [ ] `PatchInferer` — Patch-based with splitter/merger
- [ ] `SaliencyInferer` — GradCAM/CAM saliency maps
- [ ] `SliceInferer` — 2D model on 3D volumes slice-by-slice

---

## Optimizers

- [ ] `Novograd` — MONAI layer-wise adaptive optimizer
- [ ] `RMSprop`
- [ ] `LearningRateFinder` — Automatic LR range test

---

## Learning Rate Schedulers

- [ ] `WarmupCosineSchedule` — Linear warmup + cosine decay
- [ ] `CosineAnnealingWarmRestarts` — Cosine with warm restarts
- [ ] `PolynomialLR` — Polynomial decay

---

## Datasets

- [ ] `LMDBDataset` — LMDB-backed persistent cache
- [ ] `GDSDataset` — GPU Direct Storage
- [ ] `CacheNTransDataset` — Cache first N transforms
- [ ] `ArrayDataset` / `ImageDataset`
- [ ] `ZipDataset` — Combine multiple datasets
- [ ] `GridPatchDataset` / `PatchDataset`
- [ ] `ThreadDataLoader` — Faster data loading for cached data
- [ ] `DecathlonDataset` — Medical Segmentation Decathlon
- [ ] `TciaDataset`
