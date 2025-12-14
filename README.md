# Blade Defect Detection

Blade defect detection using deep learning for segmentation of defects in aero-engine blades.

## Project Description

This project implements a deep learning pipeline for detecting and segmenting defects in aero-engine blades. The model performs multi-class segmentation to identify four types of defects: dent, nick, scratch, and corrosion.

## Setup

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:
```bash
pip install uv
```

2. Clone the repository:
```bash
git clone <repository-url>
cd blade-defect-detection
```

3. Install dependencies using uv:
```bash
uv sync
```

This will:
- Create a virtual environment automatically
- Install all project dependencies
- Install development dependencies

4. Activate the virtual environment:
```bash
source .venv/bin/activate
```

Or use uv to run commands directly:
```bash
uv run <command>
```

5. Install pre-commit hooks:
```bash
uv run pre-commit install
```

6. Verify installation:
```bash
uv run pre-commit run -a
```

### Data Setup

The dataset can be organized in two ways:

**Option 1: Pre-split data (recommended)**
```
data/
├── train/
│   ├── images/
│   │   ├── normal/
│   │   ├── dent/
│   │   ├── nick/
│   │   ├── scratch/
│   │   └── corrosion/
│   └── masks/
│       ├── dent/
│       ├── nick/
│       ├── scratch/
│       └── corrosion/
└── val/
    ├── images/
    └── masks/
```

**Option 2: Single directory (auto-split 80/20)**
```
data/
├── images/
│   ├── normal/
│   ├── dent/
│   ├── nick/
│   ├── scratch/
│   └── corrosion/
└── masks/
    ├── dent/
    ├── nick/
    ├── scratch/
    └── corrosion/
```

Note: Images and masks should have matching filenames. Normal images don't require masks.

### MLflow Setup

Start MLflow server locally:
```bash
mlflow server --host 127.0.0.1 --port 8080
```

Or configure remote tracking URI in `configs/mlflow.yaml`.

## Train

### Basic Training

Train the model with default configuration:
```bash
uv run python commands.py train
```

Or if virtual environment is activated:
```bash
python commands.py train
```

### Custom Data Directory

Train with custom data directory:
```bash
uv run python commands.py train --data_dir=/path/to/data
```

### Configuration

Modify configuration files in `configs/`:
- `configs/data.yaml` - Dataset and data loader settings
- `configs/model.yaml` - Model architecture settings
- `configs/training.yaml` - Training hyperparameters
- `configs/mlflow.yaml` - MLflow tracking settings

### Training Process

The training process will:
1. Load images and masks from the data directory
2. Resize images to the configured size (default: 256x256)
3. Train a UNet model for multi-class segmentation
4. Log metrics to MLflow (loss, IoU, etc.)
5. Save the best model checkpoint

### Monitoring

**TensorBoard (recommended for real-time visualization):**
```bash
tensorboard --logdir lightning_logs --host 127.0.0.1 --port 6006
```

Then open http://127.0.0.1:6006 in your browser to view:
- Training/validation loss curves
- IoU metrics
- Learning rate
- Other training metrics

**MLflow UI:**
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Then open http://127.0.0.1:5000 in your browser.

## Project Structure

```
blade-defect-detection/
├── blade_defect_detection/    # Main package
│   ├── data/                  # Data loading
│   ├── models/                # Model definitions
│   ├── training/              # Training logic
│   └── utils/                 # Utilities
├── configs/                   # Hydra configurations
├── data/                      # Dataset (managed by DVC)
├── models/                    # Saved models (managed by DVC)
└── commands.py               # CLI entry point
```

## Dependencies

- PyTorch & PyTorch Lightning
- Hydra for configuration
- MLflow for experiment tracking
- DVC for data versioning

See `pyproject.toml` for complete dependency list.

