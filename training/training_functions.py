"""
StarDist Model Training Pipeline - Functions Module
====================================================
Contains all functions for training and fine-tuning StarDist models.

Author: Ali Attaa
Date: 02/04/2026
"""

import os
import gc
import glob
import json
import numpy as np
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Tuple, Dict, Optional, Callable
from shapely.geometry import shape
from csbdeep.utils import normalize
from stardist.models import StarDist2D, Config2D
from stardist.plot import render_label
from stardist import fill_label_holes
from augmend import Augmend, FlipRot90, Elastic, IntensityScaleShift


# ============================================================
# GPU UTILITIES
# ============================================================

def setup_gpu_environment(conda_env_path: str = None) -> None:
    """
    Setup CUDA environment for Windows.

    Args:
        conda_env_path: Path to conda environment (Windows only)
    """
    if conda_env_path and os.path.exists(conda_env_path):
        cuda_bin_path = os.path.join(conda_env_path, 'Library', 'bin')
        if os.path.exists(cuda_bin_path):
            os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ.get('PATH', '')


def check_gpu() -> bool:
    """
    Check for available GPU and configure memory growth.

    Returns:
        True if GPU is available, False otherwise
    """
    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f" Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")

        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(" Memory growth enabled")
        except RuntimeError as e:
            print(f"Could not set memory growth: {e}")

        print(f"\nTensorFlow version: {tf.__version__}")
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    else:
        print(" No GPU found - training will be slow!")

    print("=" * 60)
    return len(gpus) > 0


def clear_gpu_memory() -> None:
    """Clear TensorFlow session and run garbage collection."""
    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================
# DATA AUGMENTATION
# ============================================================

def create_augmenter() -> Callable:
    """
    Create data augmentation pipeline for training.

    Returns:
        Augmentation function compatible with StarDist training
    """
    from stardist.models import StarDist2D

    # Use StarDist's built-in augmenter instead of custom augmend
    def augmenter(x, y):
        """Apply augmentation to image and mask pair."""
        import numpy as np

        # Random flip (horizontal)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)

        # Random flip (vertical)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=0)
            y = np.flip(y, axis=0)

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        x = np.rot90(x, k, axes=(0, 1))
        y = np.rot90(y, k, axes=(0, 1))

        # Random intensity scaling (only for image, not mask)
        scale = np.random.uniform(0.8, 1.2)
        shift = np.random.uniform(-0.1, 0.1)
        x = x * scale + shift
        x = np.clip(x, 0, 1)

        # Ensure contiguous arrays
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        return x, y

    return augmenter


# ============================================================
# MASK PROCESSING
# ============================================================

def relabel_image(mask: np.ndarray) -> np.ndarray:
    """
    Relabel mask to have consecutive integer labels starting from 1.

    Args:
        mask: Input label mask

    Returns:
        Relabeled mask with consecutive labels
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    new_mask = np.zeros_like(mask)
    for new_label, old_label in enumerate(unique_labels, start=1):
        new_mask[mask == old_label] = new_label

    return new_mask


def geojson_to_instance_mask(geojson_path: str, img_shape: Tuple[int, int],
                             min_area: int = 20) -> np.ndarray:
    """
    Convert GeoJSON annotations to instance segmentation mask.

    Args:
        geojson_path: Path to GeoJSON file
        img_shape: Shape of the image (height, width) or (height, width, channels)
        min_area: Minimum nucleus area to include

    Returns:
        Instance segmentation mask
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    # Handle both (H, W) and (H, W, C) shapes
    if len(img_shape) == 3:
        height, width = img_shape[:2]
    else:
        height, width = img_shape

    mask = np.zeros((height, width), dtype=np.int32)

    # Handle different GeoJSON structures
    if isinstance(data, dict) and 'features' in data:
        features = data['features']
    elif isinstance(data, list):
        features = data
    else:
        print(f" Unknown GeoJSON structure in {geojson_path}")
        return mask

    label_id = 1
    for feature in features:
        try:
            geom = shape(feature['geometry'])

            if geom.area < min_area:
                continue

            # Get polygon coordinates
            if geom.geom_type == 'Polygon':
                coords = np.array(geom.exterior.coords).astype(np.int32)
            elif geom.geom_type == 'MultiPolygon':
                # Take the largest polygon
                largest = max(geom.geoms, key=lambda x: x.area)
                coords = np.array(largest.exterior.coords).astype(np.int32)
            else:
                continue

            # Draw filled polygon
            cv2.fillPoly(mask, [coords], label_id)
            label_id += 1

        except Exception as e:
            continue

    # Post-process mask
    mask = fill_label_holes(mask)
    mask = relabel_image(mask)

    return mask


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data(data_dir: str, img_ext: str = '.tif',
                       min_nucleus_area: int = 20,
                       normalize_images: bool = True,
                       percentile_low: float = 1.0,
                       percentile_high: float = 99.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from directory containing images and GeoJSON annotations.

    Expected file structure:
        data_dir/
            image1.tif
            image1.geojson
            image2.tif
            image2.geojson
            ...

    Args:
        data_dir: Directory containing training data
        img_ext: Image file extension
        min_nucleus_area: Minimum nucleus area to include
        normalize_images: Whether to normalize images
        percentile_low: Low percentile for normalization
        percentile_high: High percentile for normalization

    Returns:
        Tuple of (images array, masks array)
    """
    print(f"Loading training data from: {data_dir}")

    x, y = [], []

    image_paths = sorted(glob.glob(os.path.join(data_dir, f"*{img_ext}")))

    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]

        # Skip mask files
        if base.endswith("_mask"):
            continue

        geojson_path = os.path.join(data_dir, base + ".geojson")

        if not os.path.exists(geojson_path):
            print(f"  [SKIP] Missing geojson: {base}")
            continue

        # Load image
        img = imageio.imread(img_path).astype(np.float32)

        # Normalize if requested
        if normalize_images:
            img = normalize(img, percentile_low, percentile_high)

        # Load mask
        inst_mask = geojson_to_instance_mask(geojson_path, img.shape, min_nucleus_area)

        if inst_mask.max() == 0:
            print(f"  [SKIP] No nuclei in: {base}")
            continue

        x.append(img)
        y.append(inst_mask)
        print(f"  [LOADED] {base}: {inst_mask.max()} nuclei")

    x = np.array(x)
    y = np.array(y)

    print(f"\nLoaded {len(x)} training samples")
    print(f"   Images shape: {x.shape}")
    print(f"   Labels shape: {y.shape}")

    return x, y


def split_data(x: np.ndarray, y: np.ndarray,
               validation_split: float = 0.2,
               random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets.

    Args:
        x: Images array
        y: Masks array
        validation_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X_train, Y_train, X_val, Y_val)
    """
    np.random.seed(random_seed)

    n_val = max(1, int(len(x) * validation_split))
    indices = np.random.permutation(len(x))

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train, Y_train = x[train_indices], y[train_indices]
    X_val, Y_val = x[val_indices], y[val_indices]

    print(f"Data split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    return X_train, Y_train, X_val, Y_val


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_training_sample(x: np.ndarray, y: np.ndarray,
                              index: int = None,
                              save_path: str = None) -> None:
    """
    Visualize a training sample with image and labels overlay.

    Args:
        x: Images array
        y: Masks array
        index: Index of sample to visualize (random if None)
        save_path: Path to save figure (displays if None)
    """
    if index is None:
        index = np.random.randint(len(x))

    plt.figure(figsize=(12, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(x[index], 0, 1))
    plt.title(f"Training Image (index={index})")
    plt.axis("off")

    # Image with labels overlay
    plt.subplot(1, 2, 2)
    plt.imshow(render_label(y[index], img=np.clip(x[index], 0, 1)))
    plt.title(f"Instance Labels ({y[index].max()} nuclei)")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(history: Dict, save_path: str = None) -> None:
    """
    Plot training history (loss curves).

    Args:
        history: Training history dictionary
        save_path: Path to save figure (displays if None)
    """
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    if 'loss' in history:
        plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Learning rate (if available)
    plt.subplot(1, 2, 2)
    if 'lr' in history:
        plt.plot(history['lr'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to: {save_path}")
    else:
        plt.show()

    plt.close()

# ============================================================
# MODEL CREATION
# ============================================================

def create_model_from_pretrained(model_name: str, model_basedir: str,
                                 pretrained_model: str = None,
                                 finetuned_model_path: str = None,
                                 patch_size: Tuple[int, int] = (256, 256)) -> StarDist2D:
    """
    Create a new StarDist model initialized with pretrained weights.

    Args:
        model_name: Name for the new model
        model_basedir: Base directory to save new model
        pretrained_model: Name of built-in pretrained model (e.g., "2D_versatile_he")
        finetuned_model_path: Path to custom pretrained model folder (overrides pretrained_model)
        patch_size: Training patch size

    Returns:
        StarDist2D model ready for training
    """
    print(f"Creating model: {model_name}")
    print(f"  Base directory: {model_basedir}")

    os.makedirs(model_basedir, exist_ok=True)

    # Load pretrained model
    print("  Loading pretrained weights...")

    if finetuned_model_path is not None:
        # Load custom trained model
        print(f"  Custom model path: {finetuned_model_path}")
        pretrained = StarDist2D(None, name=os.path.basename(finetuned_model_path),
                                basedir=os.path.dirname(finetuned_model_path))
    else:
        # Load built-in pretrained model
        print(f"  Built-in model: {pretrained_model}")
        pretrained = StarDist2D.from_pretrained(pretrained_model)

    # Get and modify config from pretrained model
    conf = pretrained.config
    conf.train_patch_size = patch_size

    # Create new model WITH the config (not None!)
    print(f"  Creating new model with pretrained config...")
    model = StarDist2D(conf, name=model_name, basedir=model_basedir)

    # Copy pretrained weights to new model
    print(f"  Copying weights to new model...")
    model.keras_model.set_weights(pretrained.keras_model.get_weights())

    print(f"  Model created successfully")
    print(f"  Patch size: {patch_size}")

    return model


def create_model_from_pretrained_mod(model_name: str, model_basedir: str,
                                 pretrained_model: str = "2D_versatile_he",
                                 custom_model_path: str = None,
                                 n_rays: int = None,
                                 grid: Tuple[int, int] = None,
                                 patch_size: Tuple[int, int] = (256, 256),
                                 copy_weights: bool = True) -> StarDist2D:
    """
    Create a new StarDist model initialized with pretrained weights.

    Args:
        copy_weights: If False, don't copy weights (use when changing n_rays/grid)
    """
    print(f"Creating model: {model_name}")
    print(f"  Base directory: {model_basedir}")

    os.makedirs(model_basedir, exist_ok=True)

    # Load pretrained model
    print("  Loading pretrained model...")

    if custom_model_path is not None:
        print(f"  Custom model path: {custom_model_path}")
        pretrained = StarDist2D(None, name=os.path.basename(custom_model_path),
                                basedir=os.path.dirname(custom_model_path))
    else:
        print(f"  Built-in model: {pretrained_model}")
        pretrained = StarDist2D.from_pretrained(pretrained_model)

    # Get config from pretrained model
    conf = pretrained.config

    # Modify config if custom parameters provided
    if n_rays is not None:
        print(f"  Modifying n_rays: {conf.n_rays} → {n_rays}")
        conf.n_rays = n_rays
        copy_weights = False  # Can't copy weights if architecture changes

    if grid is not None:
        print(f"  Modifying grid: {conf.grid} → {grid}")
        conf.grid = grid
        copy_weights = False  # Can't copy weights if architecture changes

    # Update patch size
    conf.train_patch_size = patch_size

    # Create new model with (possibly modified) config
    print(f"  Creating new model with config...")
    model = StarDist2D(conf, name=model_name, basedir=model_basedir)

    # Copy pretrained weights ONLY if architecture is the same
    if copy_weights:
        print(f"  Copying weights from pretrained model...")
        model.keras_model.set_weights(pretrained.keras_model.get_weights())
        print(f"   Weights copied")
    else:
        print(f"  Architecture changed - weights NOT copied (training from scratch)")

    print(f"  Model created successfully")
    print(f"  n_rays: {model.config.n_rays}")
    print(f"  grid: {model.config.grid}")
    print(f"  Patch size: {patch_size}")

    return model

def create_model_from_scratch(model_name: str, model_basedir: str,
                              n_channels: int = 3,
                              patch_size: Tuple[int, int] = (256, 256),
                              n_rays: int = 96,
                              grid: Tuple[int, int] = (2, 2)) -> StarDist2D:
    """
    Create a new StarDist model from scratch.

    Args:
        model_name: Name for the new model
        model_basedir: Base directory to save model
        n_channels: Number of input channels
        patch_size: Training patch size
        n_rays: Number of radial directions
        grid: Subsampling grid

    Returns:
        StarDist2D model ready for training
    """
    print(f"Creating model from scratch: {model_name}")

    os.makedirs(model_basedir, exist_ok=True)

    conf = Config2D(
        n_channel_in=n_channels,
        n_rays=n_rays,
        grid=grid,
        use_gpu=True,
        train_patch_size=patch_size
    )

    model = StarDist2D(conf, name=model_name, basedir=model_basedir)

    print(f" Model created")
    print(f"  Channels: {n_channels}")
    print(f"  Rays: {n_rays}")
    print(f"  Grid: {grid}")
    print(f"  Patch size: {patch_size}")

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(model: StarDist2D,
                X_train: np.ndarray, Y_train: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray,
                epochs: int = 400,
                steps_per_epoch: int = 200,
                use_augmentation: bool = True) -> Dict:
    """
    Train StarDist model.

    Args:
        model: StarDist2D model to train
        X_train: Training images
        Y_train: Training masks
        X_val: Validation images
        Y_val: Validation masks
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        use_augmentation: Whether to use data augmentation

    Returns:
        Training history dictionary
    """
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"  Epochs: {epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Augmentation: {use_augmentation}")
    print("=" * 60)

    # Prepare for training
    model.prepare_for_training()

    # Create augmenter if needed
    augmenter = create_augmenter() if use_augmentation else None

    # Train
    print("\nStarting training...")
    history = model.train(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        augmenter=augmenter,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )

    print("\n Training complete")

    return history.history if hasattr(history, 'history') else {}


def optimize_thresholds(model: StarDist2D,
                        X_val: np.ndarray, Y_val: np.ndarray) -> Dict:
    """
    Optimize detection thresholds on validation data.

    Args:
        model: Trained StarDist2D model
        X_val: Validation images
        Y_val: Validation masks

    Returns:
        Optimized thresholds dictionary
    """
    print("\nOptimizing thresholds on validation data...")

    model.optimize_thresholds(X_val, Y_val)

    print(f"  Optimized thresholds: {model.thresholds}")

    return model.thresholds


def export_model(model: StarDist2D) -> None:
    """
    Export model for TensorFlow Serving deployment.

    Args:
        model: Trained StarDist2D model
    """
    print("\nExporting TensorFlow model...")

    try:
        model.export_TF()
        print(" Model exported")
    except Exception as e:
        print(f"  Export failed: {e}")


# ============================================================
# FULL TRAINING PIPELINE
# ============================================================

def run_training_pipeline(data_dir: str,
                          model_name: str,
                          model_basedir: str,
                          img_ext: str = '.tif',
                          min_nucleus_area: int = 20,
                          validation_split: float = 0.2,
                          epochs: int = 400,
                          steps_per_epoch: int = 200,
                          patch_size: Tuple[int, int] = (256, 256),
                          use_augmentation: bool = True,
                          use_pretrained: bool = True,
                          pretrained_model: str = None,
                          finetuned_model_path: str = None,
                          n_rays: int = 96,
                          grid: Tuple[int, int] = None,
                          copy_weights: bool = True,
                          visualize: bool = True,
                          export: bool = True) -> StarDist2D:
    """
    Run complete training pipeline.

    Args:
        data_dir: Directory containing training data
        model_name: Name for the new model
        model_basedir: Base directory to save model
        img_ext: Image file extension
        min_nucleus_area: Minimum nucleus area to include
        validation_split: Fraction of data for validation
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        patch_size: Training patch size
        use_augmentation: Whether to use data augmentation
        use_pretrained: Whether to start from pretrained weights
        pretrained_model: Name of pretrained model
        visualize: Whether to show visualizations
        export: Whether to export model for deployment

    Returns:
        Trained StarDist2D model
    """
    print("\n" + "=" * 70)
    print("STARDIST TRAINING PIPELINE")
    print("=" * 70)

    # Check GPU
    check_gpu()

    # Load data
    print("\n" + "-" * 50)
    print("Step 1: Loading Data")
    print("-" * 50)

    x, y = load_training_data(
        data_dir=data_dir,
        img_ext=img_ext,
        min_nucleus_area=min_nucleus_area
    )

    if len(x) == 0:
        raise RuntimeError("No valid training data found!")

    # Split data
    print("\n" + "-" * 50)
    print("Step 2: Splitting Data")
    print("-" * 50)

    X_train, Y_train, X_val, Y_val = split_data(
        x, y,
        validation_split=validation_split
    )

    # Visualize sample
    if visualize:
        print("\n" + "-" * 50)
        print("Step 3: Visualizing Sample")
        print("-" * 50)

        vis_path = os.path.join(model_basedir, model_name, "training_sample.png")
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        visualize_training_sample(X_train, Y_train, save_path=vis_path)

    # Create model
    print("\n" + "-" * 50)
    print("Step 4: Creating Model")
    print("-" * 50)

    if use_pretrained:
        model = create_model_from_pretrained_mod(
            model_name=model_name,
            model_basedir=model_basedir,
            pretrained_model=pretrained_model,
            n_rays=n_rays,
            grid=grid,
            patch_size=patch_size,
            copy_weights=copy_weights
        )
    else:
        model = create_model_from_scratch(
            model_name=model_name,
            model_basedir=model_basedir,
            patch_size=patch_size,
            n_rays=n_rays
        )

    # Train
    print("\n" + "-" * 50)
    print("Step 5: Training Model")
    print("-" * 50)

    history = train_model(
        model=model,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        use_augmentation=use_augmentation
    )

    # Plot training history
    if visualize and history:
        history_path = os.path.join(model_basedir, model_name, "training_history.png")
        plot_training_history(history, save_path=history_path)

    # Optimize thresholds
    print("\n" + "-" * 50)
    print("Step 6: Optimizing Thresholds")
    print("-" * 50)

    optimize_thresholds(model, X_val, Y_val)

    # Export model
    if export:
        print("\n" + "-" * 50)
        print("Step 7: Exporting Model")
        print("-" * 50)

        export_model(model)

    # Print final info
    model_dir = os.path.join(model_basedir, model_name)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n Model saved to: {model_dir}")
    print("\nModel files:")

    if os.path.exists(model_dir):
        for f in sorted(os.listdir(model_dir)):
            print(f"   - {f}")

    print("=" * 70)

    return model