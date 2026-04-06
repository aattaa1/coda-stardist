
# StarDist Nuclear Segmentation Pipeline

[![Python 3.10.19](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31019/)
[![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10-orange.svg)](https://www.tensorflow.org/)

A streamlined, GPU-accelerated pipeline for nuclear segmentation of Whole Slide Images (WSIs) and image tiles using [StarDist](https://github.com/stardist/stardist). Designed for computational pathology workflows with outputs compatible with [QuPath](https://qupath.github.io/) and MATLAB.

---

## Features

- **Efficient Segmentation**: Uses StarDist's `predict_instances_big()` for memory-efficient processing of large WSIs
- **Automatic Method Selection**: Intelligently switches between methods based on image size
- **Multiple Input Formats**: Supports NDPI, SVS, TIFF, PNG, and JPEG files
- **Comprehensive Feature Extraction**: Extracts 26 nuclear morphology and intensity features
- **QuPath Integration**: Outputs GeoJSON files with proper labeling for QuPath visualization
- **MATLAB Compatibility**: Exports features as MAT files with pixel resolution metadata
- **GPU Accelerated**: Full CUDA support with automatic fallback to CPU
- **Modular Design**: Use individual functions or run the complete pipeline

---

## Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Segmentation Pipeline](#segmentation-pipeline)
  - [Training Pipeline](#training-pipeline)
  - [Programmatic Usage](#programmatic-usage)
- [Output Files](#-output-files)
- [Features Extracted](#-features-extracted)
- [Configuration Options](#-configuration-options)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Driver 450.x or higher

### Step 1: Clone the Repository or Download As zip folder

```bash
git clone https://github.com/YOUR_USERNAME/coda-stardist.git
cd coda-stardist
```

or

1. Click the green "<> Code" button
2. Click "Download ZIP"

### Step 2: Create Conda Environment

The `environment.yml` file contains all dependencies including CUDA, TensorFlow, and StarDist:

```bash
conda env create -f environment.yml
```

This will:
- Create a new environment called `coda-stardist`
- Install Python 3.10
- Install CUDA 11.8 and cuDNN 8.9
- Install TensorFlow, StarDist, and all other dependencies

### Step 3: Activate Environment

```bash
conda activate coda-stardist
```

### Step 4: Verify Installation

```bash
python utilities/verify_installation.py
```

Expected output:
```
============================================================
STARDIST PIPELINE - INSTALLATION VERIFICATION
============================================================

✅ Python 3.10.x
✅ TensorFlow: 2.10.0
✅ GPU detected: 1 device(s)
✅ StarDist: 0.9.2
✅ OpenSlide: 4.0.0
...
✅ ALL CHECKS PASSED!
```

---

## Quick Start

### 1. Segment Images

```bash
# Activate environment
conda activate coda-stardist

# Edit configuration with your paths
# Open apply_stardist.py and set:
#   WSI_PATHS = [r'path/to/your/images']
#   MODEL_PATH = r'path/to/your/model'

# Run segmentation
python apply_stardist.py
```

### 2. Train a Model (Optional)

```bash
# Edit configuration with your paths
# Open train_stardist.py and set:
#   DATA_DIR = r'path/to/training/data'
#   MODEL_NAME = 'My_Model'

# Run training
python train_stardist.py
```

---

## Usage

### Segmentation Pipeline

#### Configuration

Edit `apply_stardist.py`:

```python
# Input directories containing WSIs or image tiles
WSI_PATHS = [
    r'Z:\Path\To\Your\Images',
    r'Z:\Another\Image\Directory',
]

# Path to trained StarDist model
MODEL_PATH = r"Z:\Path\To\Your\Model"
MODEL_NAME = "HCC_HE_Finetuned"

# Processing options
EXTRACT_RGB_FEATURES = True   # Extract RGB intensity features
GENERATE_GEOJSON = True       # Generate QuPath-compatible files
GEOJSON_EVERY_N = 50          # Generate GeoJSON every N images
```

#### Run

```bash
conda activate stardist_pipeline
python apply_stardist.py
```

#### Output Structure

```
Your_Image_Directory/
├── image1.ndpi
├── image2.ndpi
│
├── StarDist_##_##_202#_ModelName/
│   ├── geojsons/
│   │   └── imageN.geojson      # Every Nth image
│   ├── feature_pickles/
│   │   ├── image1.pkl           # Features + pixel resolution for python inference
│   │   └── image2.pkl
│   └── feature_mat/
│       ├── image1.mat           # Features + pixel resolution for matlab inference
│       └── image2.mat

```

---

### Training Pipeline

#### Data Preparation

Organize your training data:

```
training_data/
├── image_001.tif
├── image_001.geojson    # Polygon annotations
├── image_002.tif
├── image_002.geojson
└── ...
```

GeoJSON annotation format:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[x1,y1], [x2,y2], ..., [x1,y1]]]
      }
    }
  ]
}
```

#### Configuration

Edit `train_stardist.py`:

```python
DATA_DIR = r"Z:\Path\To\Training\Data"
MODEL_NAME = "My_Custom_Model"
MODEL_BASEDIR = r"Z:\Path\To\Save\Models"

EPOCHS = 400
STEPS_PER_EPOCH = 200
VALIDATION_SPLIT = 0.2
USE_PRETRAINED = True  # Start from pretrained weights (recommended)
```

#### Run

```bash
conda activate stardist_pipeline
python train_stardist.py
```

---

### Programmatic Usage

```python
from segmentation.stardist_functions import (
    check_gpu,
    load_model,
    segment_image,
    extract_features_from_polys,
    save_geojson_from_polys,
    process_single_image,
    process_image_directory
)

# Check GPU
check_gpu()

# Load model
model = load_model("path/to/model")

# Segment single image
mask, polys = segment_image("path/to/image.ndpi", model)

# Extract features
df = extract_features_from_polys(polys, "path/to/image.ndpi", extract_rgb=True)

# Save GeoJSON
save_geojson_from_polys(polys, "output/dir", "image_name")

# Or process entire directory
stats = process_image_directory(
    wsi_dir="path/to/images",
    model=model,
    output_base="path/to/output"
)
```

---

## Output Files

| File Type | Extension | Description | Software |
|-----------|-----------|-------------|----------|
| GeoJSON | `.geojson` | Nuclear contours with labels | QuPath, Python |
| Pickle | `.pkl` | Feature DataFrame | Python, Pandas |
| MAT Binaries | `.mat` | Features + pixel resolution | MATLAB, Python |

### Loading Output Files

**Python (Pickle):**
```python
import pandas as pd
df = pd.read_pickle("image.pkl")
print(df.head())
```

**Python (MAT):**
```python
from scipy.io import loadmat
data = loadmat("image.mat")
features = data['features']
feature_names = data['feature_names']
pix_res = data['pix_res']
num_nuclei = data['num_nuclei']
```

**MATLAB:**
```matlab
data = load('image.mat');
features = data.features;
feature_names = data.feature_names;
pix_res = data.pix_res;
num_nuclei = data.num_nuclei;
```

**QuPath:**
1. Open WSI in QuPath
2. File → Import → Import Objects
3. Select `.geojson` file
4. Nuclei will appear with labels 1, 2, 3...

---

## 📊 Features Extracted

The pipeline extracts **26 nuclear morphology and intensity features** for each detected nucleus.

### Feature Categories

| Category | Features | Count |
|----------|----------|-------|
| **Position** | Centroid_x, Centroid_y | 2 |
| **Size** | Area, Perimeter, convex_area, equiv_diameter | 4 |
| **Shape** | Circularity, Aspect_Ratio, compactness, eccentricity, extent, form_factor, solidity | 7 |
| **Radius** | maximum_radius, mean_radius, median_radius | 3 |
| **Axis** | minor_axis_length, major_axis_length, orientation_degrees | 3 |
| **Intensity** | r_mean_intensity, g_mean_intensity, b_mean_intensity, r_std, g_std, b_std | 6 |
| **Metadata** | slide_num, label_id | 2 |

---

### Shape Feature Equations

#### Circularity

Measures how close the shape is to a perfect circle.

$$\text{Circularity} = \frac{4\pi \cdot A}{P^2}$$

- Range: $(0, 1]$
- Value of $1$ indicates a perfect circle

---

#### Solidity

Measures the proportion of the convex hull filled by the object.

$$\text{Solidity} = \frac{A}{A_{convex}}$$

- Range: $(0, 1]$
- Value of $1$ indicates a convex shape

---

#### Eccentricity

Measures how elongated the shape is.

$$\text{Eccentricity} = \sqrt{1 - \left(\frac{b}{a}\right)^2}$$

- Range: $[0, 1)$
- Value of $0$ indicates a circle
- Value approaching $1$ indicates a line

---

#### Compactness

Inverse measure of how compact the shape is.

$$\text{Compactness} = \frac{P^2}{A}$$

- Minimum value of $4\pi \approx 12.57$ for a circle
- Higher values indicate less compact shapes

---

#### Form Factor

Normalized measure of shape complexity (inverse of circularity).

$$\text{Form Factor} = \frac{P^2}{4\pi \cdot A}$$

- Range: $[1, \infty)$
- Value of $1$ indicates a perfect circle

---

#### Equivalent Diameter

Diameter of a circle with the same area as the object.

$$\text{Equiv Diameter} = \sqrt{\frac{4A}{\pi}}$$

---

#### Extent

Ratio of object area to the area of its bounding ellipse.

$$\text{Extent} = \frac{A}{a \cdot b}$$

- Range: $(0, 1]$

---

#### Aspect Ratio

Ratio of major to minor axis length.

$$\text{Aspect Ratio} = \frac{a}{b}$$

- Range: $[1, \infty)$
- Value of $1$ indicates equal axes (circular)

---

### Radius Features

For each contour point $p_i$ and centroid $c$:

$$d_i = \|p_i - c\|_2$$

| Feature | Formula |
|---------|---------|
| **Maximum Radius** | $\max(d_1, d_2, ..., d_n)$ |
| **Mean Radius** | $\frac{1}{n}\sum_{i=1}^{n} d_i$ |
| **Median Radius** | $\text{median}(d_1, d_2, ..., d_n)$ |

---

### Intensity Features

For pixels $\{p_1, p_2, ..., p_n\}$ within the nucleus mask:

| Feature | Formula |
|---------|---------|
| **Mean Intensity** | $\mu_c = \frac{1}{n}\sum_{i=1}^{n} p_i^{(c)}$ |
| **Std Intensity** | $\sigma_c = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (p_i^{(c)} - \mu_c)^2}$ |

Where $c \in \{R, G, B\}$ represents the color channel.

---

### Variable Definitions

| Symbol | Definition |
|--------|------------|
| $A$ | Area of the nucleus (pixels²) |
| $P$ | Perimeter of the nucleus (pixels) |
| $A_{convex}$ | Area of the convex hull (pixels²) |
| $a$ | Major axis length (pixels) |
| $b$ | Minor axis length (pixels) |
| $c$ | Centroid coordinates |
| $n$ | Number of contour points or pixels |

---

## ⚙️ Configuration Options

### Segmentation Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `WSI_PATHS` | List of input directories | Required |
| `MODEL_PATH` | Path to StarDist model | Required |
| `MODEL_NAME` | Model name for output folder | Required |
| `EXTRACT_RGB_FEATURES` | Extract intensity features | `True` |
| `GENERATE_GEOJSON` | Create QuPath files | `True` |
| `GEOJSON_EVERY_N` | GeoJSON frequency | `50` |
| `BLOCK_SIZE` | Tile size for large images | `4096` |
| `INVERSE_ORDER` | Process in reverse | `False` |

### Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DATA_DIR` | Training data directory | Required |
| `MODEL_NAME` | Name for new model | Required |
| `MODEL_BASEDIR` | Directory to save model | Required |
| `EPOCHS` | Training epochs | `400` |
| `STEPS_PER_EPOCH` | Steps per epoch | `200` |
| `VALIDATION_SPLIT` | Validation fraction | `0.2` |
| `USE_PRETRAINED` | Use pretrained weights | `True` |
| `PATCH_SIZE` | Training patch size | `(256, 256)` |
| `USE_AUGMENTATION` | Enable data augmentation | `True` |

---

## Project Structure

```
stardist-pipeline/
│
├── README.md                 # This file
|── train_stardist.py       # Training execution script
|── apply_stardist.py       # Segmentation execution script
├── environment.yml           # Conda environment (all dependencies)
├── requirements.txt          # Pip dependencies (alternative)
│
├── segmentation/
│   ├── stardist_functions.py # Core segmentation functions
│  
│
├── training/
│   ├── stardist_training.py  # Core training functions
│   
│
├── utilities/
│   └── verify_installation.py # Installation checker
│
└── docs/
    └── images/               # Documentation images
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA in TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Windows Solution:** Add CUDA to path in your script:
```python
import os
CONDA_ENV = r"C:\Users\USERNAME\miniconda3\envs\stardist_pipeline"
os.environ['PATH'] = os.path.join(CONDA_ENV, 'Library', 'bin') + os.pathsep + os.environ['PATH']
```

### Out of Memory Error

**Solution:** Reduce block size in `run_pipeline.py`:
```python
BLOCK_SIZE = 2048  # Reduce from 4096
```

### OpenSlide Error

**Solution:** Reinstall OpenSlide packages:
```bash
conda activate coda-stardist
pip uninstall openslide-python openslide-bin
pip install openslide-bin openslide-python
```

### Environment Creation Fails

**Solution:** Try creating with explicit channel priority:
```bash
conda env create -f environment.yml --force
```

Or create manually:
```bash
conda create -n stardist_pipeline python=3.10 -y
conda activate stardist_pipeline
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9 -y
pip install -r requirements.txt
```

### Small Images Failing

The pipeline automatically handles small images (<2000×2000 pixels) by using `predict_instances()` instead of `predict_instances_big()`. No action needed.

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{stardist_pipeline,
  author = {Ali Attaa},
  title = {StarDist Nuclear Segmentation Pipeline},
  year = {2026},
  url = {https://github.com/aattaa1/coda-stardist}
}
```

Also cite the original StarDist papers:

```bibtex
@inproceedings{schmidt2018,
  author    = {Uwe Schmidt and Martin Weigert and Coleman Broaddus and Gene Myers},
  title     = {Cell Detection with Star-Convex Polygons},
  booktitle = {Medical Image Computing and Computer Assisted Intervention - {MICCAI} 2018},
  year      = {2018}
}

@inproceedings{weigert2020,
  author    = {Martin Weigert and Uwe Schmidt and Robert Haase and Ko Sugawara and Gene Myers},
  title     = {Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2020}
}
```

---

## Acknowledgments

- [StarDist](https://github.com/stardist/stardist) - Original segmentation method
- [CSBDeep](https://github.com/CSBDeep/CSBDeep) - Deep learning framework
- [QuPath](https://qupath.github.io/) - Whole slide image analysis
- [OpenSlide](https://openslide.org/) - WSI file reading
- [TensorFlow](https://www.tensorflow.org/) - Deep learning platform

---

## Contact

**Your Name**  
Email: aattaa1@jh.edu  
Lab: [Kiemen Lab](https://labs.pathology.jhu.edu/kiemen/)

---

<p align="center">
  Made with 💛 for computational pathology purposes
</p>
