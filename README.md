# Blade Defect Detection

Deep learning pipeline for detecting and segmenting defects in aero-engine blades using multi-class semantic segmentation.

## Context

This project is conceptually based on the paper  
**"BladeSynth: A High-Quality Rendering-Based Synthetic Dataset for Aero Engine Blade Defect Inspection"**  
(*Scientific Data*, 2025).  
DOI: https://doi.org/10.1038/s41597-025-05563-y

The paper introduces a rendering-based synthetic data generation pipeline for industrial defect inspection. Instead of data-driven generative models, defects are created using explicit analytical and procedural formulations applied to CAD-based aero-engine blade geometry.

### Key Concepts from the Paper

- **Mapping 3D blade surfaces to 2D texture space** using UV unwrapping
- **Procedural defect modeling** via parametric height maps:
  - **Dents** modeled as quadratic surfaces with depth parameters and noise
  - **Scratches** modeled using linear/quadratic profiles with absolute terms
  - **Corrosion** generated using stochastic noise fields affecting color and roughness maps
  - **Nicks** generated through volumetric Boolean subtraction of perturbed geometric primitives
- Use of spatial masks and noise functions to introduce realistic surface irregularities
- **Domain randomization** over camera pose, lighting conditions, materials, backgrounds, and defect parameters
- **Physically Based Rendering (PBR)** to produce photorealistic RGB images paired with pixel-level segmentation masks

The paper demonstrates that models trained solely on the generated synthetic data generalize to real-world inspection images (sim-to-real transfer), making this approach suitable as a reference for reproducible and scalable ML/MLOps pipelines based on synthetic data.

## Project Description

This project implements a complete MLOps pipeline for automated defect detection in aero-engine turbine blades. The system uses deep learning to perform pixel-level semantic segmentation, identifying and localizing four types of surface defects:

- **Dent** - surface dents and deformations
- **Nick** - small cuts or nicks in the blade edge
- **Scratch** - surface scratches and abrasions
- **Corrosion** - corrosion damage and oxidation

### Problem Statement

Manual inspection of turbine blades is time-consuming, subjective, and prone to human error. This project automates the detection and classification of blade defects through computer vision, enabling:
- Faster quality control in manufacturing
- Consistent defect classification
- Precise localization of defects for repair planning
- Automated documentation of blade condition

### Technical Approach

The solution uses a **UNet architecture** for semantic segmentation, which:
- Takes RGB images of blades as input (configurable size, default: 192×192 pixels)
- Original dataset images are 1024×1024 pixels (from BladeSynth paper)
- Outputs pixel-level class predictions (background + 4 defect types)
- Provides precise spatial localization of defects
- Handles multi-class segmentation with skip connections for detail preservation

### Model Architecture

The model implements a **UNet** architecture with:
- **Encoder**: 4 downsampling blocks (40→80→160→320 channels) with MaxPooling
- **Bottleneck**: 640-channel bottleneck layer
- **Decoder**: 4 upsampling blocks with skip connections from encoder
- **Output**: 5-class segmentation map (background + 4 defect types)

Each block uses double convolutions (Conv2d + BatchNorm + ReLU) for feature extraction.

### Key Technologies

- **PyTorch Lightning** - Training framework with automatic GPU support and checkpointing
- **DVC** - Data version control with Yandex Object Storage (public HTTP access) for 27GB dataset
- **Hydra** - Hierarchical configuration management for hyperparameters
- **MLflow** - Experiment tracking with metrics, hyperparameters, and git commit logging
- **Ruff** - Fast Python linter and formatter (replaces black, isort, flake8)

## Features

- ✅ **Automatic data management**: DVC integration with automatic pull on train/predict
- ✅ **Flexible configuration**: Hydra-based hierarchical configs for all hyperparameters
- ✅ **Comprehensive logging**: MLflow (metrics, hyperparameters, git commit id)
- ✅ **Model versioning**: Automatic checkpointing with best model selection
- ✅ **Inference pipeline**: Single-command prediction with visualization
- ✅ **Code quality**: Pre-commit hooks with ruff, prettier, and code quality checks
- ✅ **Production-ready**: Error handling, graceful fallbacks, and comprehensive documentation

## Setup

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) for dependency management
- CUDA-capable GPU (recommended for training)

### Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:
```bash
pip install uv
```

2. **Clone the repository**:
```bash
git clone https://github.com/sidorov-ka/blade-defect-detection.git
cd blade-defect-detection
```

3. **Install dependencies**:
```bash
uv sync
```

This will:
- Create a virtual environment automatically (`.venv/`)
- Install all project dependencies
- Install development dependencies (ruff, pre-commit, pytest)

4. **Activate the virtual environment** (optional):
```bash
source .venv/bin/activate
```

Or use `uv run` to execute commands directly:
```bash
uv run python commands.py train
```

5. **Install pre-commit hooks**:
```bash
uv run pre-commit install
```

6. **Verify installation**:
```bash
uv run pre-commit run -a
```

### Environment Configuration

1. **Create `.env` file** (copy from `env.example`):
```bash
cp env.example .env
```

2. **Configure environment variables** (optional) in `.env`:
```env
# MLflow Configuration (optional)
# MLflow server should be running on this URI for experiment tracking
MLFLOW_TRACKING_URI=http://127.0.0.1:8080

# Data Directory (optional, defaults to 'data')
DATA_DIR=data
```

**Note**: 
- `.env` is already in `.gitignore`
- DVC uses public bucket access, no credentials needed for pulling data

### DVC Setup

Data is managed by DVC and stored in Yandex Object Storage (public bucket). The DVC remote is already configured for public HTTP access - no credentials needed.

**First time setup** (if data is not available locally):
```bash
# Pull data from cloud storage (public access, no credentials needed)
uv run dvc pull
```

**Note**: 
- Data will be automatically pulled when running `train` command if not available locally
- Progress indicator will show during download (may take several minutes for ~30GB dataset)
- If you experience timeout errors during `dvc pull`, you can increase timeouts:
  ```bash
  uv run dvc remote modify storage read_timeout 600
  uv run dvc remote modify storage connect_timeout 180
  ```

## Data Structure

The dataset should be organized as follows:

```
data/
├── images/
│   ├── normal/          # Normal blades (no defects)
│   ├── dent/            # Images with dents
│   ├── nick/            # Images with nicks
│   ├── scratch/         # Images with scratches
│   └── corrosion/       # Images with corrosion
└── masks/
    ├── dent/            # Masks for dent defects
    ├── nick/            # Masks for nick defects
    ├── scratch/         # Masks for scratch defects
    └── corrosion/       # Masks for corrosion defects
```

**Important**:
- Images and masks should have matching filenames
- Normal images don't require masks (all background)
- Masks should be binary (0=background, 255=defect) or normalized (0-1)
- Some files in the original dataset may have prefixes like `valid_` in their names (from pre-split validation sets)

**Dataset Splitting**:
- If `data/train/`, `data/val/`, and `data/test/` directories exist, they will be used directly
- Otherwise, the code automatically splits the dataset 70/15/15 (train/val/test) using a fixed random seed (42)
- When using automatic splitting, files with prefixes (like `valid_`) may appear in any split, but their original names are preserved

### Dataset Statistics

- **Total size**: ~27 GB
- **Total files**: ~45,000 images
- **Original image resolution**: 1024×1024 pixels (from BladeSynth paper)
- **Training image size**: Configurable via `configs/data/dataset.yaml` (default: 192×192)
- **Classes**: 4 defect types + background
- **Format**: PNG images with corresponding mask files
- **Storage**: Yandex Object Storage (S3-compatible) via DVC
- **Type**: Synthetic data generated using BladeSynth pipeline (rendering-based)

### Dataset Generation Method

The dataset is generated using the BladeSynth pipeline, which:

1. **Maps 3D blade CAD models to 2D texture space** using UV unwrapping
2. **Applies procedural defect models** to create realistic surface defects:
   - Dents: Quadratic surfaces with depth and noise parameters
   - Scratches: Linear/quadratic profiles with absolute terms
   - Corrosion: Stochastic noise fields affecting color and roughness
   - Nicks: Volumetric Boolean operations on geometric primitives
3. **Renders photorealistic images** using Physically Based Rendering (PBR)
4. **Generates pixel-level masks** automatically during rendering
5. **Applies domain randomization** to camera, lighting, materials, and defect parameters

This synthetic approach enables training models that generalize to real-world inspection scenarios, as demonstrated in the original paper.

## Usage

### Training

**Basic training** with default configuration:
```bash
uv run python commands.py train
```

**Custom data directory**:
```bash
uv run python commands.py train --data_dir=/path/to/data
```

**What happens during training**:
1. DVC automatically checks for data in `data/` directory
2. If data is missing, it pulls from cloud storage (or restores from cache)
3. Loads hierarchical configuration from `configs/` using Hydra
4. Creates train/val/test splits (70/15/15) if not pre-split
5. Trains UNet model for multi-class segmentation with PyTorch Lightning
6. Saves best checkpoint (lowest validation loss) to `models/best.ckpt`
7. Logs metrics, hyperparameters, and git commit id to MLflow

**Training outputs**:
- **Model checkpoints**: `models/best-*.ckpt` (best model), `models/last*.ckpt` (last checkpoint)
- **MLflow experiments**: `blade-defect-detection` experiment with all runs

**Logged metrics**:
- `train_loss` - CrossEntropy loss (training only)
- `val_loss`, `val_loss_ce`, `val_loss_dice` - Combined loss (0.5 * CrossEntropy + 0.5 * Dice) and components
- `test_loss` - Combined loss (0.5 * CrossEntropy + 0.5 * Dice)
- `train_iou`, `val_iou`, `test_iou` - Jaccard Index (IoU) for segmentation quality
- Hyperparameters: learning_rate, batch_size, num_epochs, image_size, etc.
- Git commit ID as tag for reproducibility

### Inference (Prediction)

**Predict on a single image**:
```bash
uv run python commands.py predict \
    --image_path=path/to/image.png \
    --output_path=visualizations/prediction.png
```

**Use specific checkpoint**:
```bash
uv run python commands.py predict \
    --image_path=path/to/image.png \
    --checkpoint_path=models/best-epoch=09-val_loss=0.06.ckpt
```

**What happens during prediction**:
1. Loads model from checkpoint (or latest from `models/`)
2. Preprocesses input image
3. Runs inference
4. Creates visualization with predicted mask overlay
5. Saves result to `visualizations/` (or specified path)

### Monitoring Training

**MLflow UI** (if MLflow server is running):
```bash
uv run mlflow ui --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 in your browser to view:
- All training runs with metrics comparison
- Hyperparameter search results
- Git commit IDs for each run
- Model artifacts and checkpoints

**Start MLflow server** (required before training):
```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

**Note**: The training code automatically detects if MLflow server is available. If not reachable, training continues without MLflow logging (graceful fallback).

## Configuration

All hyperparameters and settings are managed through **Hydra** hierarchical configuration files in `configs/`:

### Configuration Structure

- `configs/config.yaml` - Main configuration file (composes all sub-configs)
- `configs/data/dataset.yaml` - Dataset settings:
  - Image size: `[192, 192]` (height, width) - resized from original 1024×1024
  - Defect classes: `[dent, nick, scratch, corrosion]`
  - Batch size: `6` (with gradient accumulation, effective batch size: 36)
  - Number of workers: `2`
- `configs/model/model.yaml` - Model architecture:
  - Input channels: `3` (RGB)
  - Number of classes: `5` (background + 4 defects)
- `configs/training/training.yaml` - Training hyperparameters:
  - Number of epochs: `10`
  - Learning rate: `0.0001`
  - Weight decay: `0.00001`
  - Optimizer: `Adam`
  - Learning rate scheduler: `ReduceLROnPlateau` (patience=3, factor=0.5)
  - Early stopping: Enabled (patience=5, min_delta=0.001)
  - Data augmentation: ColorJitter (brightness, contrast, saturation, hue)
  - Class weights: Automatically computed from training data
  - Precision: 32-bit (full precision)
  - Gradient accumulation: 6 batches (effective batch size: 36)
  - Gradient clipping: 1.0
- `configs/mlflow/mlflow.yaml` - MLflow tracking:
  - Tracking URI: `http://127.0.0.1:8080`
  - Experiment name: `blade-defect-detection`
  - Run name: auto-generated if null

### Modifying Configuration

**Option 1: Edit YAML files directly**
```bash
# Edit configs/training/training.yaml
num_epochs: 50
learning_rate: 0.001
```

**Option 2: Override via command line** (recommended for experiments)
```bash
uv run python commands.py train \
    training.training.num_epochs=50 \
    training.training.learning_rate=0.001 \
    data.dataloader.batch_size=16
```

**Option 3: Use Hydra's multi-run** (for hyperparameter search)
```bash
uv run python commands.py train \
    training.training.learning_rate=0.0001,0.001,0.01 \
    -m
```

### No Magic Constants

All hyperparameters are in configuration files. No hardcoded values in the code.

## Project Structure

```
blade-defect-detection/
├── blade_defect_detection/     # Main Python package
│   ├── data/                    # Dataset loading (BladeDefectDataset)
│   ├── models/                  # Model definitions (UNet)
│   ├── training/                # Training logic (Lightning module)
│   └── utils/                   # Utilities (logging, etc.)
├── configs/                     # Hydra configuration files
│   ├── config.yaml              # Main config
│   ├── data/                    # Data configuration
│   ├── model/                   # Model configuration
│   ├── training/                # Training configuration
│   └── mlflow/                  # MLflow configuration
├── data/                        # Dataset directory (managed by DVC)
│   └── README.md                # Data directory info
├── models/                      # Saved model checkpoints
│   └── .gitkeep                 # Keep directory in Git
├── visualizations/              # Prediction visualizations
│   └── .gitkeep                 # Keep directory in Git
├── .dvc/                        # DVC configuration
│   ├── config                   # DVC remote settings
│   └── cache/                   # Local DVC cache (28GB)
├── commands.py                  # CLI entry point (train, predict)
├── pyproject.toml               # Project dependencies and metadata
├── .pre-commit-config.yaml      # Pre-commit hooks configuration
├── .gitignore                   # Git ignore rules
├── .env                         # Environment variables (not in Git)
└── README.md                    # This file
```

## DVC Workflow

### Data Management

**Pull data from cloud** (if not available locally):
```bash
# Public bucket access, no credentials needed
uv run dvc pull
```

**Add new data version**:
```bash
# After modifying data/
uv run dvc add data/
uv run dvc push
git add data.dvc
git commit -m "Update dataset"
```

**Check DVC status**:
```bash
uv run dvc status
```

**Restore data from cache**:
```bash
uv run dvc checkout data.dvc
```

### Model Versioning (Optional)

You can also version models with DVC:
```bash
# Add models to DVC
uv run dvc add models/

# Push to cloud
uv run dvc push

# Commit metadata
git add models.dvc
git commit -m "Add trained model"
```

## Dependencies

Dependencies are managed using **uv** (modern Python package manager) and defined in `pyproject.toml`.

### Main Dependencies

**Deep Learning & Training**:
- `pytorch-lightning>=2.0.0` - Training framework with automatic GPU support
- `torchvision>=0.15.0` - Image transforms and utilities
- `torchmetrics>=1.0.0` - Metrics (JaccardIndex/IoU for segmentation)

**Configuration & CLI**:
- `hydra-core>=1.3.0` - Hierarchical configuration management
- `omegaconf>=2.3.0` - Configuration format (used by Hydra)
- `fire>=0.5.0` - CLI framework (replaces argparse)

**Data Management**:
- `dvc>=3.0.0` - Data version control
- `dvc-s3>=3.0.0` - S3-compatible storage support (for Yandex Object Storage)

**Experiment Tracking**:
- `mlflow>=2.8.0` - Experiment tracking and model registry

**Utilities**:
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
- `requests>=2.31.0` - HTTP requests (for MLflow health checks)

### Development Dependencies

- `ruff>=0.1.0` - Fast linter and formatter (replaces black, isort, flake8)
- `pre-commit>=3.0.0` - Git hooks for code quality
- `pytest>=7.4.0` - Testing framework
- `ipykernel>=6.0.0` - Jupyter notebook support

### Installation

All dependencies are installed automatically with:
```bash
uv sync
```

This creates a virtual environment (`.venv/`) and installs all dependencies from `pyproject.toml` and `uv.lock`.

## Development

### Code Quality Tools

The project uses **pre-commit** hooks with the following tools:

- **Ruff** - Fast Python linter and formatter (replaces black, isort, flake8)
- **Prettier** - Formatter for YAML, JSON, TOML, Markdown files
- **Pre-commit hooks** - Basic checks (YAML, JSON, large files, trailing whitespace, etc.)
- **Codespell** - Spell checker for code

**Configuration**: All tools are configured in:
- `pyproject.toml` - Ruff settings (line length, target version, rules)
- `.pre-commit-config.yaml` - Pre-commit hooks configuration

**Run code quality checks**:
```bash
# Run all pre-commit hooks
uv run pre-commit run -a

# Or run individual tools
uv run ruff check .          # Lint
uv run ruff format .         # Format
```

**Install pre-commit hooks** (runs automatically on git commit):
```bash
uv run pre-commit install
```

### Testing

**Run tests** (when available):
```bash
uv run pytest
```

### Code Style

- **File naming**: `snake_case` for Python files (e.g., `dataset.py`, not `Dataset.py`)
- **Package naming**: `snake_case` for package directory (`blade_defect_detection`)
- **Line length**: 100 characters (configured in `pyproject.toml`)
- **Import style**: Organized with ruff (isort replacement)
- **No magic constants**: All values in Hydra configs or constants files

## Troubleshooting

### DVC Issues

**Problem**: `dvc pull` fails
**Solution**: Check internet connection. The bucket is public and doesn't require credentials.

**Problem**: Data not found locally
**Solution**: Run `uv run dvc pull` or data will be automatically pulled when running `train`

### Training Issues

**Problem**: CUDA out of memory
**Solution**: Reduce `batch_size` in `configs/data/dataset.yaml` or image size

**Problem**: MLflow connection timeout or server not running
**Solution**: The code automatically detects MLflow server availability with a 3-second timeout. If unreachable, training continues without MLflow logging (graceful fallback). Start MLflow server before training: `uv run mlflow server --host 127.0.0.1 --port 8080`

### Data Issues

**Problem**: Invalid mask values error
**Solution**: Ensure masks are binary (0-255) or normalized (0-1), and class indices are in range [0, num_classes-1]

## Metrics and Evaluation

### Training Metrics

The model is evaluated using:

- **Loss Functions**:
  - **Training**: CrossEntropyLoss only (for speed)
  - **Validation/Test**: Combined loss (0.5 * CrossEntropy + 0.5 * Dice Loss)
  - Dice Loss directly optimizes segmentation quality (IoU)
  - Class weights are applied to CrossEntropyLoss to handle imbalanced data
- **IoU (Jaccard Index)**: Macro-averaged IoU across all classes
  - Measures overlap between predicted and ground truth masks
  - Range: [0, 1], higher is better
  - Logged separately for train/val/test sets

### Test Set

The test set is either:
- Pre-split: Uses `data/test/` directory if available
- Auto-split: 15% of the full dataset (70% train, 15% val, 15% test)

Test metrics are logged after training completes and can be viewed in MLflow.

## Reproducibility

- **Git commit ID**: Automatically logged to MLflow as a tag for each training run
- **Random seeds**: Fixed seed (42) for dataset splitting
- **Configuration**: All hyperparameters stored in Hydra configs
- **Data versioning**: DVC tracks dataset versions via `data.dvc`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Sidorov Konstantin** - <kasidorov@gmail.com>

## Acknowledgments

- **BladeSynth Dataset**: This project uses the synthetic dataset generation pipeline from the paper "BladeSynth: A High-Quality Rendering-Based Synthetic Dataset for Aero Engine Blade Defect Inspection" (*Scientific Data*, 2025). DOI: https://doi.org/10.1038/s41597-025-05563-y
- **PyTorch Lightning** team for the excellent training framework
- **DVC** team for data versioning tools
- **Hydra** team for configuration management
- **UNet architecture**: Original paper by Ronneberger et al. (2015)
