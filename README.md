# Blade Defect Detection

Deep learning pipeline for detecting and segmenting defects in aero-engine blades using multi-class semantic segmentation.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Project Structure](#project-structure)
7. [Troubleshooting](#troubleshooting)

---

## Requirements

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows (WSL2)
- **GPU**: CUDA-capable GPU (recommended for training)
- **Disk Space**: At least 30 GB free space for dataset

### Software Requirements

- **uv**: Modern Python package manager
- **Git**: For cloning the repository
- **MLflow Server** (optional): For experiment tracking

---

## Installation

### Step 1: Install uv

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (pip):**
```bash
pip install uv
```

### Step 2: Clone Repository

```bash
git clone https://github.com/sidorov-ka/blade-defect-detection.git
cd blade-defect-detection
```

### Step 3: Install Dependencies

```bash
uv sync
```

This command will:
- Create virtual environment in `.venv/`
- Install all project dependencies
- Install development dependencies (ruff, pre-commit, pytest)

### Step 4: Install Pre-commit Hooks (Optional)

```bash
uv run pre-commit install
```

### Step 5: Verify Installation

```bash
uv run python -c "from blade_defect_detection.data.dataset import BladeDefectDataset; print('Installation successful')"
```

---

## Quick Start

### 1. Start MLflow Server (Optional)

Open a new terminal and run:
```bash
cd blade-defect-detection
uv run mlflow server --host 127.0.0.1 --port 8080
```

Keep this terminal open during training.

### 2. Run Training

```bash
uv run python commands.py train
```

The script will:
- Automatically download data via DVC if not available locally
- Load configuration from `configs/`
- Split dataset into train/val/test (70/15/15)
- Train UNet model for 10 epochs
- Save best checkpoint to `models/`
- Log metrics to MLflow (if server is running)

### 3. Run Prediction

```bash
uv run python commands.py predict \
    --image_path=path/to/image.png \
    --output_path=visualizations/prediction.png
```

---

## Usage

### Training

#### Basic Training

```bash
uv run python commands.py train
```

#### Custom Data Directory

```bash
uv run python commands.py train --data_dir=/path/to/data
```

#### Override Configuration Parameters

```bash
uv run python commands.py train \
    training.training.num_epochs=50 \
    training.training.learning_rate=0.001 \
    data.dataloader.batch_size=16
```

#### Hyperparameter Search (Multi-run)

```bash
uv run python commands.py train \
    training.training.learning_rate=0.0001,0.001,0.01 \
    -m
```

### Prediction

#### Basic Prediction

```bash
uv run python commands.py predict --image_path=path/to/image.png
```

#### Use Specific Checkpoint

```bash
uv run python commands.py predict \
    --image_path=path/to/image.png \
    --checkpoint_path=models/best-epoch=09-val_loss=0.06.ckpt \
    --output_path=visualizations/result.png
```

### Data Management

#### Pull Data from DVC

```bash
uv run dvc pull
```

#### Check DVC Status

```bash
uv run dvc status
```

#### Restore Data from Cache

```bash
uv run dvc checkout data.dvc
```

---

## Configuration

### Configuration Files Location

All configuration files are located in `configs/` directory:

```
configs/
├── config.yaml              # Main configuration
├── data/
│   └── dataset.yaml         # Dataset settings
├── model/
│   └── model.yaml           # Model architecture
├── training/
│   └── training.yaml       # Training hyperparameters
└── mlflow/
    └── mlflow.yaml          # MLflow settings
```

### Default Configuration

**Dataset** (`configs/data/dataset.yaml`):
- Image size: `192×192` pixels
- Batch size: `6` (effective: 36 with gradient accumulation)
- Number of workers: `2`
- Defect classes: `[dent, nick, scratch, corrosion]`

**Training** (`configs/training/training.yaml`):
- Number of epochs: `10`
- Learning rate: `0.0001`
- Weight decay: `0.00001`
- Optimizer: `Adam`
- Learning rate scheduler: `ReduceLROnPlateau` (patience=3, factor=0.5)
- Early stopping: Enabled (patience=5, min_delta=0.001)
- Gradient accumulation: `6` batches
- Gradient clipping: `1.0`

**Model** (`configs/model/model.yaml`):
- Input channels: `3` (RGB)
- Number of classes: `5` (background + 4 defects)

**MLflow** (`configs/mlflow/mlflow.yaml`):
- Tracking URI: `http://127.0.0.1:8080`
- Experiment name: `blade-defect-detection`

### Modifying Configuration

**Method 1: Edit YAML Files**

Edit configuration files directly:
```bash
# Edit configs/training/training.yaml
num_epochs: 50
learning_rate: 0.001
```

**Method 2: Command Line Override** (Recommended)

Override parameters via command line:
```bash
uv run python commands.py train \
    training.training.num_epochs=50 \
    training.training.learning_rate=0.001
```

**Method 3: Environment Variables**

Set environment variables (optional):
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
export DATA_DIR=data
```

---

## Project Structure

```
blade-defect-detection/
├── blade_defect_detection/      # Main Python package
│   ├── data/                     # Dataset loading
│   │   ├── __init__.py
│   │   └── dataset.py            # BladeDefectDataset class
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   └── model.py              # UNet architecture
│   ├── training/                 # Training logic
│   │   ├── __init__.py
│   │   └── train.py              # Lightning module and training
│   ├── inference/               # Inference pipeline
│   │   ├── __init__.py
│   │   └── predict.py            # Prediction functions
│   └── utils/                    # Utilities
│       └── logging.py            # MLflow logging utilities
├── configs/                      # Hydra configuration files
│   ├── config.yaml              # Main config
│   ├── data/
│   │   └── dataset.yaml         # Dataset configuration
│   ├── model/
│   │   └── model.yaml             # Model configuration
│   ├── training/
│   │   └── training.yaml        # Training configuration
│   └── mlflow/
│       └── mlflow.yaml          # MLflow configuration
├── data/                         # Dataset directory (DVC managed)
├── models/                       # Saved model checkpoints
├── visualizations/              # Prediction visualizations
├── commands.py                   # CLI entry point
├── pyproject.toml                # Project dependencies
├── data.dvc                      # DVC data tracking file
└── README.md                     # This file
```

### Data Structure

The dataset should be organized as follows:

```
data/
├── images/
│   ├── normal/                  # Normal blades (no defects)
│   ├── dent/                     # Images with dents
│   ├── nick/                     # Images with nicks
│   ├── scratch/                  # Images with scratches
│   └── corrosion/                # Images with corrosion
└── masks/
    ├── dent/                     # Masks for dent defects
    ├── nick/                     # Masks for nick defects
    ├── scratch/                  # Masks for scratch defects
    └── corrosion/               # Masks for corrosion defects
```

**Important Notes:**
- Images and masks must have matching filenames
- Normal images don't require masks (treated as all background)
- Masks should be binary (0=background, 255=defect) or normalized (0-1)
- If `data/train/`, `data/val/`, and `data/test/` directories exist, they will be used directly
- Otherwise, dataset is automatically split 70/15/15 (train/val/test) with seed=42

---

## Troubleshooting

### Installation Issues

**Problem**: `uv: command not found`
```bash
# Solution: Install uv (see Installation Step 1)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Problem**: `ModuleNotFoundError` after installation
```bash
# Solution: Ensure you're using uv run
uv run python commands.py train
# Or activate virtual environment
source .venv/bin/activate
python commands.py train
```

### DVC Issues

**Problem**: `dvc pull` fails
```bash
# Solution 1: Check internet connection
ping storage.yandexcloud.net

# Solution 2: Retry with verbose output
uv run dvc pull -v

# Solution 3: Data will be automatically pulled when running train command
```

**Problem**: Data not found locally
```bash
# Solution: Pull data manually
uv run dvc pull
# Or data will be automatically pulled when running train
```

### Training Issues

**Problem**: CUDA out of memory
```bash
# Solution 1: Reduce batch size in configs/data/dataset.yaml
batch_size: 4  # or lower

# Solution 2: Reduce image size in configs/data/dataset.yaml
image_size: [128, 128]  # instead of [192, 192]

# Solution 3: Reduce gradient accumulation in configs/training/training.yaml
# (Note: This requires modifying the code, as it's hardcoded)
```

**Problem**: MLflow connection errors
```bash
# Solution 1: Start MLflow server before training
uv run mlflow server --host 127.0.0.1 --port 8080

# Solution 2: Training will continue without MLflow if server is unavailable
# (This is expected behavior - graceful fallback)
```

**Problem**: Training stops early
```bash
# Check: Early stopping is enabled by default (patience=5)
# This is expected if validation loss doesn't improve
# Adjust in configs/training/training.yaml if needed
```

### Data Issues

**Problem**: `No samples found in data_dir`
```bash
# Solution 1: Ensure data is downloaded
uv run dvc pull

# Solution 2: Check data directory structure matches expected format
ls -R data/images/
ls -R data/masks/

# Solution 3: Verify image extensions are supported (.jpg, .jpeg, .png, .bmp, .tiff, .tif)
```

**Problem**: `Invalid mask values error`
```bash
# Solution: Ensure masks are:
# - Binary format (0=background, 255=defect) OR normalized (0-1)
# - Class indices in range [0, num_classes-1]
# - Same filename as corresponding image
```

### MLflow Issues

**Problem**: MLflow server returns 500 errors
```bash
# Solution 1: Restart MLflow server
# Stop current server (Ctrl+C) and restart:
uv run mlflow server --host 127.0.0.1 --port 8080

# Solution 2: Delete and recreate MLflow database
rm mlflow.db
uv run mlflow server --host 127.0.0.1 --port 8080

# Solution 3: Training continues without MLflow (graceful fallback)
```

---

## Additional Information

### Dataset Information

- **Total size**: ~27 GB
- **Total files**: ~45,000 images
- **Original resolution**: 1024×1024 pixels
- **Training resolution**: 192×192 pixels (configurable)
- **Classes**: 4 defect types + background
- **Format**: PNG images with corresponding mask files
- **Storage**: Yandex Object Storage (public access via DVC)

### Model Architecture

- **Type**: UNet for semantic segmentation
- **Encoder**: 4 downsampling blocks (40→80→160→320 channels)
- **Bottleneck**: 640-channel layer
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 5-class segmentation map

### Training Metrics

- **Loss Functions**:
  - Training: CrossEntropyLoss (with class weights)
  - Validation/Test: 0.5 × CrossEntropyLoss + 0.5 × DiceLoss
- **Metrics**: IoU (Jaccard Index) - macro-averaged across all classes
- **Checkpoints**: Best model (lowest validation loss) saved to `models/`

### Reproducibility

- **Random seed**: Fixed to 42 for dataset splitting
- **Git commit ID**: Automatically logged to MLflow (if available)
- **Configuration**: All hyperparameters in Hydra configs
- **Data versioning**: DVC tracks dataset versions

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Sidorov Konstantin** - <kasidorov@gmail.com>

## Acknowledgments

- **BladeSynth Dataset**: Based on "BladeSynth: A High-Quality Rendering-Based Synthetic Dataset for Aero Engine Blade Defect Inspection" (*Scientific Data*, 2025)
- **PyTorch Lightning** - Training framework
- **DVC** - Data version control
- **Hydra** - Configuration management
- **UNet**: Original architecture by Ronneberger et al. (2015)
