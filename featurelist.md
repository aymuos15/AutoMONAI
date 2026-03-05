# AutoMONAI Feature Tracker

Features from MONAI not yet supported. Check off as implemented.

---

## Models / Architectures

### Segmentation
- [x] `BasicUNet` — Simplified U-Net
- [x] `BasicUNetPlusPlus` — UNet++ with nested skip connections
- [x] `DynUNet` — Dynamic U-Net (nnU-Net style)
- [x] `VNet` — Volumetric segmentation
- [x] `HighResNet` — Brain parcellation
- [x] `UNETR` — UNet with Vision Transformers
- [x] `SegResNetVAE` — SegResNet with variational autoencoder
- [x] `SegResNetDS` — SegResNet with deep supervision
- [x] `SegResNetDS2` — SegResNet DS variant 2
- [x] `FlexibleUNet` — Configurable backbone U-Net
- [x] `DiNTS` — Differentiable Neural Architecture Search
- [x] `MedNeXt` (S/M/B/L) — ConvNeXt-style medical nets


---

## Loss Functions

### Segmentation Losses
- [x] `DiceCELoss` — Dice + Cross Entropy combined
- [x] `DiceFocalLoss` — Dice + Focal combined
- [x] `GeneralizedDiceLoss` — Weighted Dice for class imbalance
- [x] `GeneralizedWassersteinDiceLoss` — Wasserstein distance-based
- [x] `GeneralizedDiceFocalLoss` — Generalized Dice + Focal
- [x] `TverskyLoss` — Alpha/beta weighted Dice variant
- [x] `HausdorffDTLoss` — Hausdorff Distance Transform loss
- [x] `LogHausdorffDTLoss` — Log Hausdorff DT loss
- [x] `SoftclDiceLoss` — Soft centerline Dice (tubular structures)
- [x] `SoftDiceclDiceLoss` — Soft Dice + clDice combined
- [x] `MaskedDiceLoss` — Spatially masked Dice
- [x] `NACLLoss` — Neighbor-Aware Constrained Loss
- [x] `AsymmetricUnifiedFocalLoss` — Asymmetric focal
- [x] `SSIMLoss` — Structural similarity
- [ ] `PerceptualLoss` — LPIPS-based
- [x] `DeepSupervisionLoss` — Multi-scale weighted loss


### Loss Wrappers
- [ ] `MaskedLoss` — Spatial mask wrapper
- [ ] `MultiScaleLoss` — Multi-scale smoothing wrapper

---

## Transforms

### Spatial
- [ ] `Spacing` / `Spacingd` — Resample to target voxel spacing
- [ ] `Orientation` / `Orientationd` — Canonical axis reorientation
- [x] `RandAffine` — Random affine (combined rotate/scale/translate)
- [x] `RandElasticDeformation` / `Rand2DElastic` / `Rand3DElastic` — Elastic deformation
- [x] `RandCropByPosNegLabel` — Balanced foreground/background patch sampling
- [x] `CropForeground` — Auto-crop to foreground
- [x] `RandRotate90` — Random 90° rotation
- [x] `RandSpatialCropSamples` — Multiple random crop samples
- [x] `SpatialPad` / `BorderPad` / `DivisiblePad` — Padding transforms
- [x] `GridPatch` / `GridSplit` — Grid-based patching

### Intensity
- [x] `RandGibbsNoise` — Gibbs artifact augmentation
- [x] `RandKSpaceSpikeNoise` — K-space spike noise
- [x] `RandBiasField` — Bias field augmentation
- [x] `RandCoarseDropout` — Cutout augmentation
- [x] `RandCoarseShuffle` — Coarse shuffle
- [x] `RandHistogramShift` — Histogram shifting
- [x] `RandStdShiftIntensity` / `RandShiftIntensity` — Intensity shifting
- [x] `RandScaleIntensity` — Random intensity scaling
- [x] `RandGaussianSmooth` — Gaussian smoothing
- [x] `RandGaussianSharpen` — Gaussian sharpening
- [ ] `GaussianSmooth` / `GaussianSharpen` — Deterministic smoothing/sharpening
- [x] `MaskIntensity` — Mask-based intensity filtering
- [x] `ClipIntensityPercentiles` — Percentile-based clipping
- [x] `ScaleIntensityRange` — Range-based intensity scaling
- [x] `ThresholdIntensity` — Intensity thresholding

---

## Metrics

- [x] `HausdorffDistanceMetric` — Boundary distance
- [x] `SurfaceDistanceMetric` — Average surface distance
- [x] `SurfaceDiceMetric` — Normalized surface Dice (NSD)
- [x] `GeneralizedDiceScore` — Weighted Dice score
- [x] `ConfusionMatrixMetric` — Sensitivity, specificity, precision, F1
- [ ] `ROCAUCMetric` — AUC-ROC
- [x] `FBetaScore` — F-beta score
- [x] `PanopticQualityMetric` — PQ, SQ, RQ
- [ ] `CalibrationMetric` — Expected calibration error


---

## Inferers

- [x] `SlidingWindowInferer` — Sliding window with overlap/blending
- [x] `PatchInferer` — Patch-based with splitter/merger
- [x] `SaliencyInferer` — GradCAM/CAM saliency maps
- [x] `SliceInferer` — 2D model on 3D volumes slice-by-slice

---

## Optimizers

- [x] `Novograd` — MONAI layer-wise adaptive optimizer
- [x] `RMSprop`
- [ ] `LearningRateFinder` — Automatic LR range test

---

## Learning Rate Schedulers

- [x] `WarmupCosineSchedule` — Linear warmup + cosine decay
- [x] `CosineAnnealingWarmRestarts` — Cosine with warm restarts
- [x] `PolynomialLR` — Polynomial decay

---

## Datasets

- [x] `LMDBDataset` — LMDB-backed persistent cache
- [ ] `GDSDataset` — GPU Direct Storage
- [x] `CacheNTransDataset` — Cache first N transforms
- [x] `ArrayDataset` / `ImageDataset`
- [x] `ZipDataset` — Combine multiple datasets
- [x] `GridPatchDataset` / `PatchDataset`
- [ ] `ThreadDataLoader` — Faster data loading for cached data
- [x] `DecathlonDataset` — Medical Segmentation Decathlon
- [ ] `TciaDataset`
