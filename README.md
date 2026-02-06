# coda-stardist
Train and Apply StarDist nuclear segmentation; streamlined pipeline for nuclear segmentation of Whole Slide Images (WSIs) and image tiles using StarDist.
# StarDist Nuclear Segmentation Pipeline

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A streamlined, GPU-accelerated pipeline for nuclear segmentation of Whole Slide Images (WSIs) and image tiles using [StarDist](https://github.com/stardist/stardist). Designed for computational pathology workflows with outputs compatible with [QuPath](https://qupath.github.io/) and MATLAB.

<p align="center">
  <img src="docs/images/pipeline_overview.png" alt="Pipeline Overview" width="800"/>
</p>

---

## Features

- **Efficient Segmentation**: Uses StarDist's `predict_instances_big()` for memory-efficient processing of large WSIs
- **Automatic Method Selection**: Intelligently switches between methods based on image size
- **Multiple Input Formats**: Supports NDPI, SVS, TIFF, PNG, and JPEG files
- **Comprehensive Feature Extraction**: Extracts 26 nuclear morphology and intensity features
- **QuPath Integration**: Outputs GeoJSON files with labeling for QuPath visualization
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
- [Output Files](#-output-files)
- [Features Extracted](#-features-extracted)
- [Configuration Options](#-configuration-options)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## Installation

  ### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 
- NVIDIA GPU with CUDA support (Recommended RTX 4060 with 8GB of DRAM or higher)
- NVIDIA Driver 450.x or higher

  ### Step 1: Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/stardist-pipeline.git
cd stardist-pipeline
```

### Step 2: Create Conda Environment

The environment.yml file contains all dependencies including CUDA, TensorFlow, and StarDist:

```bash
conda env create -f environment.yml
```

This will:
Create a new environment called coda-stardist
Install Python 3.10
Install CUDA 11.8 and cuDNN 8.9
Install TensorFlow, StarDist, and all other dependencies

### Step 3: Activate Environment

```bash
conda activate stardist_pipeline
```

### Step 4: Verify Installation

```bash
python utilities/verify_installation.py
============================================================
STARDIST CODA - INSTALLATION VERIFICATION
============================================================

✅ Python 3.10.x
✅ TensorFlow: 2.10.0
✅ GPU detected: 1 device(s)
✅ StarDist: 0.9.2
✅ OpenSlide: 4.0.0
...
✅ ALL CHECKS PASSED!

```
## Quick Start

### 1. Segment Images
```bash
# Activate environment
conda activate stardist_pipeline

# Edit configuration with your paths
# Open segmentation/run_pipeline.py and set:
#   WSI_PATHS = [r'path/to/your/images']
#   MODEL_PATH = r'path/to/your/model'

# Run segmentation
python segmentation/run_pipeline.py

```
### 2. Train a model

```bash
# Edit configuration with your paths
# Open training/run_training.py and set:
#   DATA_DIR = r'path/to/training/data'
#   MODEL_NAME = 'My_Model'

# Run training
python training/run_training.py

```
## Usage

### Segmentation
#### Configuration
Edit segmentation/run_pipeline.py:
```python
# Input directories containing WSIs or image tiles
WSI_PATHS = [
    r'Z:\Path\To\Your\Images',
    r'Z:\Another\Image\Directory',
]

# Path to trained StarDist model
MODEL_PATH = r"Z:\Path\To\Your\Model"
MODEL_NAME = "HCC_HE_Finetuned" #your_model

# Processing options
EXTRACT_RGB_FEATURES = True   # Extract RGB intensity features
GENERATE_GEOJSON = True       # Generate QuPath-compatible files
GEOJSON_EVERY_N = 50          # Generate GeoJSON every N images
```

Then run in your IDE or in a CLI as such:
```bash
conda activate stardist_pipeline
python segmentation/run_pipeline.py

```
Your output structure should be something like this:

```python
Your_Image_Directory/
├── image1.ndpi
├── image2.ndpi
│
├── StarDist_02_06_2026_ModelName/
│   ├── geojsons/
│   │   └── image50.geojson      # Every Nth image
│   ├── feature_pickles/
│   │   ├── image1.pkl           # Every image
│   │   └── image2.pkl
│   └── feature_mat/
        ├── image1.mat           # Features + pixel resolution
        └── image2.mat
```
### Training

####Data Preparation

Organize your training data:
```python
training_data/
├── image_001.tif
├── image_001.geojson    # Polygon annotations
├── image_002.tif
├── image_002.geojson
└── ...
```
Configuration
Edit training/run_training.py:
```python
DATA_DIR = r"Z:\Path\To\Training\Data"
MODEL_NAME = "My_Custom_Model"
MODEL_BASEDIR = r"Z:\Path\To\Save\Models"

EPOCHS = 400
STEPS_PER_EPOCH = 200
VALIDATION_SPLIT = 0.2
USE_PRETRAINED = True  # Start from pretrained weights (recommended)
```
Then run in your IDE or in a CLI as such:
```bash
conda activate stardist_pipeline
python training/run_training.py
```
## Output Files
| File Type | Description | Software |
|---|---|---|
| GeoJSON (.geojson) | Nuclear Countors and Labels | QuPath |
| Pickle (.pkl) | Nuclear Features Dataframe | Python/IDE |
| Matlab Binaries (.mat) | Nuclear Features + Pixel Resolutions | MATLAB/Python |

### Loading Output Files

### Python Pickle files:

```python
import pandas as pd
df = pd.read_pickle("image.pkl")
print(df.head())
```

### Python MAT files:
```python
from scipy.io import loadmat
data = loadmat("image.mat")
features = data['features']
feature_names = data['feature_names']
pix_res = data['pix_res']
num_nuclei = data['num_nuclei']
```
### MATLAB MAT files:
```matlab
data = load('image.mat');
features = data.features;
feature_names = data.feature_names;
pix_res = data.pix_res;
num_nuclei = data.num_nuclei;
```
### GeoJSONs and QuPath
1. Open WSI in QuPath
2. Drag and drop .geojson file on top of WSI
- Note: Be sure to be zoomed in on a certain location when you load the annotations to avoid lagging and loading of all nuclei annotations.
## Nuclear Features Explained

The pipeline extracts 26 nuclear features:

| Category | Features |
|---|---|
| **Position** | Centroid_x, Centroid_y | 
| **Size** | Area, Perimeter, convex_area, equiv_diameter |
| **Shape** | Circularity, Aspect_Ratio, compactness, eccentricity, extent, form_factor, solidity |
| **Radius** | maximum_radius, mean_radius, median_radius	 |
| **Axis** | 	minor_axis_length, major_axis_length, orientation_degrees |
| **Intensity** | r_mean_intensity, g_mean_intensity, b_mean_intensity, r_std, g_std, b_std	 |
| **Metadata** | slide_num |



### Feature Definitions

| Feature | Formula | Description |
|---------|---------|-------------|
| **Circularity** | $\dfrac{4\pi \cdot A}{P^2}$ | 1 = perfect circle |
| **Solidity** | $\dfrac{A}{A_{convex}}$ | 1 = convex shape |
| **Eccentricity** | $\sqrt{1 - \left(\dfrac{b}{a}\right)^2}$ | 0 = circle, 1 = line |
| **Compactness** | $\dfrac{P^2}{A}$ | Lower = more compact |
| **Form Factor** | $\dfrac{P^2}{4\pi \cdot A}$ | 1 = circle |
| **Equiv Diameter** | $\sqrt{\dfrac{4A}{\pi}}$ | Diameter of equivalent circle |
| **Extent** | $\dfrac{A}{a \cdot b}$ | Bounding ellipse fill ratio |
| **Aspect Ratio** | $\dfrac{a}{b}$ | Major to minor axis ratio |

Where:
- $A$ = Area
- $P$ = Perimeter
- $A_{convex}$ = Convex hull area
- $a$ = Major axis length
- $b$ = Minor axis length

### Configuration Options


