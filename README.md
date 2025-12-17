# Blade Defect Detection

Deep learning pipeline for detecting and segmenting defects in aero-engine blades using multi-class semantic segmentation.

## Project Description

This project implements a complete MLOps pipeline for blade defect detection. The model performs multi-class segmentation to identify four types of defects:
- **Dent** - surface dents
- **Nick** - small cuts or nicks
- **Scratch** - surface scratches
- **Corrosion** - corrosion damage

The project uses:
- **PyTorch Lightning** for training framework
- **DVC** for data versioning with Yandex Object Storage (S3-compatible)
- **Hydra** for configuration management
- **MLflow** for experiment tracking
- **TensorBoard** for real-time monitoring

## Features

- ✅ Automatic data loading from DVC (cloud storage)
- ✅ Configurable training pipeline with Hydra
- ✅ Experiment tracking with MLflow and TensorBoard
- ✅ Model checkpointing and versioning
- ✅ Inference pipeline with visualization
- ✅ Code quality tools (ruff, pre-commit)

## Setup

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) for dependency management
- CUDA-capable GPU (recommended for training)
- Yandex.Cloud account with Object Storage bucket (for DVC)

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

2. **Configure S3 credentials** for DVC in `.env`:
```env
# Yandex Object Storage (S3-compatible) credentials for DVC
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key

# MLflow Configuration (optional)
MLFLOW_TRACKING_URI=http://127.0.0.1:8080

# Data Directory (optional, defaults to 'data')
DATA_DIR=data
```

**Note**: `.env` is already in `.gitignore` - your credentials won't be committed.

### DVC Setup

Data is managed by DVC and stored in Yandex Object Storage. The DVC remote is already configured.

**First time setup** (if data is not available locally):
```bash
# Load environment variables
export $(cat .env | grep AWS | xargs)

# Pull data from cloud storage
uv run dvc pull
```

**Note**: Data will be automatically pulled when running `train` command if not available locally.

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

The dataset can also be pre-split into train/val/test directories (see `configs/data/dataset.yaml`).

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
3. Loads configuration from `configs/`
4. Trains UNet model for multi-class segmentation
5. Saves best checkpoint to `models/`
6. Logs metrics to MLflow and TensorBoard

**Training outputs**:
- Model checkpoints: `models/best-epoch=XX-val_loss=X.XX.ckpt`
- TensorBoard logs: `lightning_logs/version_X/`
- MLflow experiments: configured in `configs/mlflow/mlflow.yaml`

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

**TensorBoard** (recommended for real-time monitoring):
```bash
tensorboard --logdir lightning_logs --host 127.0.0.1 --port 6006
```

Open http://127.0.0.1:6006 in your browser to view:
- Training/validation loss curves
- IoU metrics per class
- Learning rate schedule
- Other training metrics

**MLflow UI** (if MLflow server is running):
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 in your browser.

**Start MLflow server** (if needed):
```bash
mlflow server --host 127.0.0.1 --port 8080
```

## Configuration

All hyperparameters and settings are managed through Hydra configuration files in `configs/`:

- `configs/config.yaml` - Main configuration file
- `configs/data/dataset.yaml` - Dataset settings (image size, classes, batch size)
- `configs/model/model.yaml` - Model architecture settings
- `configs/training/training.yaml` - Training hyperparameters (epochs, learning rate)
- `configs/mlflow/mlflow.yaml` - MLflow tracking settings

**Modify configuration**:
Edit the respective YAML files or override via command line:
```bash
uv run python commands.py train training.training.num_epochs=100
```

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
├── lightning_logs/              # TensorBoard logs
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
export $(cat .env | grep AWS | xargs)
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

Main dependencies:
- `pytorch-lightning>=2.0.0` - Training framework
- `torchvision>=0.15.0` - Image transforms
- `torchmetrics>=1.0.0` - Metrics (IoU)
- `hydra-core>=1.3.0` - Configuration management
- `omegaconf>=2.3.0` - Configuration format
- `mlflow>=2.8.0` - Experiment tracking
- `dvc>=3.0.0` - Data version control
- `dvc-s3>=3.0.0` - S3 support for DVC
- `fire>=0.5.0` - CLI framework
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
- `tensorboard>=2.20.0` - Visualization
- `requests>=2.31.0` - HTTP requests

Development dependencies:
- `ruff>=0.1.0` - Linter and formatter
- `pre-commit>=3.0.0` - Git hooks
- `pytest>=7.4.0` - Testing framework
- `ipykernel>=6.0.0` - Jupyter support

See `pyproject.toml` for complete dependency list.

## Development

### Code Quality

**Run linter and formatter**:
```bash
uv run ruff check .
uv run ruff format .
```

**Run pre-commit hooks**:
```bash
uv run pre-commit run -a
```

### Testing

**Run tests** (when available):
```bash
uv run pytest
```

## Troubleshooting

### DVC Issues

**Problem**: `dvc pull` fails with authentication error
**Solution**: Check `.env` file has correct `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

**Problem**: Data not found locally
**Solution**: Run `uv run dvc pull` or data will be automatically pulled when running `train`

### Training Issues

**Problem**: CUDA out of memory
**Solution**: Reduce `batch_size` in `configs/data/dataset.yaml` or image size

**Problem**: MLflow connection timeout
**Solution**: The code handles MLflow failures gracefully - training will continue with TensorBoard only

### Data Issues

**Problem**: Invalid mask values error
**Solution**: Ensure masks are binary (0-255) or normalized (0-1), and class indices are in range [0, num_classes-1]

## License

[Add your license here]

## Authors

[Add your name and contact information]

## Acknowledgments

- PyTorch Lightning team for the excellent training framework
- DVC team for data versioning tools
- Hydra team for configuration management
