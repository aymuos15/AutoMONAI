# MonaiUI Test Suite

Comprehensive test suite for all medical image segmentation models (UNet, Attention UNet, SegResNet, and SwinUNETR).

## Test Coverage

### test_models.py

#### TestModelInstantiation
- Tests that all models can be created with various configurations
- Tests 2D and 3D spatial dimensions
- Tests different numbers of input/output channels
- Tests different image sizes

#### TestForwardPass
- Tests 2D forward passes with correct output shapes
- Tests 3D forward passes with correct output shapes
- Tests all models individually
- Verifies output shapes match expected dimensions

#### TestTrainingIntegration
- Tests complete training steps with DiceLoss
- Tests with dummy datasets for both 2D and 3D
- Tests gradient computation and backpropagation
- Tests with PyTorch DataLoaders

#### TestModelComparison
- Tests inference consistency across all models
- Verifies models have learnable parameters
- Tests with different output channel sizes
- Checks for NaN/Inf values in outputs

#### TestErrorHandling
- Tests invalid model names
- Tests invalid channel configurations
- Tests error handling for edge cases

## Running Tests

### Install test dependencies
```bash
pip install pytest pytest-cov
```

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_models.py
```

### Run specific test class
```bash
pytest tests/test_models.py::TestModelInstantiation
```

### Run specific test
```bash
pytest tests/test_models.py::TestModelInstantiation::test_model_creation
```

### Run with coverage report
```bash
pytest --cov=src tests/
```

### Run tests excluding CUDA tests (CPU only)
```bash
pytest -m "not cuda"
```

### Run with verbose output
```bash
pytest -v
```

### Run with detailed output on failures
```bash
pytest -vv --tb=long
```

## Test Models

The test suite covers the following models:
- **UNet**: Standard U-Net architecture
- **Attention UNet**: U-Net with attention gates
- **SegResNet**: Segmentation ResNet
- **SwinUNETR**: Swin Transformer-based UNETR

## Test Configurations

### Spatial Dimensions
- 2D: Tests with (B, C, H, W) tensors
- 3D: Tests with (B, C, D, H, W) tensors

### Image Sizes
- Small: 64×64 or 64×64×64
- Medium: 128×128 or 128×128×128
- Large: 256×256 (2D only due to memory)

### Channel Configurations
- Input channels: 1, 3
- Output channels: 2, 3, 4, 5, 10

## Expected Test Results

All tests should pass on systems with:
- PyTorch 2.0+
- MONAI 1.0+
- CUDA 11.0+ (optional for GPU testing)

## Notes

- Tests with dummy data are used to verify functionality
- Real dataset tests should be run separately with `python3 -m src.run`
- CUDA tests will be skipped on CPU-only systems
- Some tests use random data, so slight variations in loss values are expected
