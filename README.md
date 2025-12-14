# Blade Defect Detection

Blade defect detection using deep learning for segmentation of defects in aero-engine blades.

## Project Description

This project implements a deep learning pipeline for detecting and segmenting defects in aero-engine blades. The model performs multi-class segmentation to identify four types of defects: dent, nick, scratch, and corrosion.

## Setup

### Prerequisites

- Python 3.9+
- Poetry for dependency management

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd blade-defect-detection
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

Or if you prefer to activate manually:
```bash
source $(poetry env info --path)/bin/activate
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

5. Verify installation:
```bash
pre-commit run -a
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
python commands.py train
```

### Custom Data Directory

Train with custom data directory:
```bash
python commands.py train --data_dir=/path/to/data
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

