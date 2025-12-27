# Data Directory

This directory contains the training dataset for blade defect detection.

## Structure

The dataset is organized as follows:

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

## Dataset Information

- **Total size**: ~27 GB
- **Total files**: ~45,000 images
- **Original resolution**: 1024×1024 pixels
- **Training resolution**: 192×192 pixels (configurable)
- **Classes**: 4 defect types + background
- **Format**: PNG images with corresponding mask files

## Data Management

This directory is managed by **DVC** (Data Version Control) and is not tracked by Git.

### Downloading Data

Data is automatically downloaded when running the training command:

```bash
uv run python commands.py train
```

Or manually:

```bash
uv run dvc pull
```

### Data Requirements

- **Images**: Must have matching filenames with corresponding masks
- **Normal images**: Don't require masks (treated as all background)
- **Masks**: Should be binary (0=background, 255=defect) or normalized (0-1)
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

### Dataset Splitting

- If `data/train/`, `data/val/`, and `data/test/` directories exist, they will be used directly
- Otherwise, the dataset is automatically split 70/15/15 (train/val/test) with random seed 42

## Storage

Data is stored in **Yandex Object Storage** (S3-compatible) via DVC with public HTTP access. No credentials are required for downloading.

## Source

This dataset is based on the **BladeSynth** synthetic data generation pipeline from the paper:
"BladeSynth: A High-Quality Rendering-Based Synthetic Dataset for Aero Engine Blade Defect Inspection" (*Scientific Data*, 2025)

DOIs: https://doi.org/10.1038/s41597-025-05563-y
