"""
StarDist Model Training - Main Execution Script
================================================
Run this script to train or fine-tune a StarDist model.

Usage:
    python train_stardist.py

Author: Ali Attaa
Date: 02/04/2026
"""

from training.training_functions import (
    setup_gpu_environment,
    check_gpu,
    run_training_pipeline,
    clear_gpu_memory
)
import os

from stardist.models import StarDist2D, Config2D
# Set environment variable before importing TensorFlow
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import training functions


# ============================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================

# GPU Environment (Windows only - set to your conda env path)
CONDA_ENV_PATH = r"C:\Users\aattaa1\AppData\Local\miniconda3\envs\coda-stardist"

# Training data directory
DATA_DIR = r"Z:\Yu Shen\Wen-Chen Chen\nuclear segmentation\qupath annotations from AK\Qupath project\ground_truth"

# Model output settings
MODEL_NAME = "wcc_02_24_2026_96_nrays"
MODEL_BASEDIR = r"\\10.99.134.183\kiemen-lab-data\Ali Attaa\nuclear segmentation\stardist models"

# Data settings
IMG_EXT = ".tif"
MIN_NUCLEUS_AREA = 20
VALIDATION_SPLIT = 0.2

# Training settings
EPOCHS = 450
STEPS_PER_EPOCH = 200
PATCH_SIZE = (256, 256)
USE_AUGMENTATION = True

# Model parameters (None = use pretrained defaults)
N_RAYS = 96  # Change to 96, 64, 32, etc. (None = use pretrained)
GRID = (2, 2)  # Change if needed (None = use pretrained)
COPY_WEIGHTS = False
USE_PRETRAINED = True

# Set FINETUNED_MODEL_PATH = None to use built-in model
PRETRAINED_MODEL = '2D_versatile_he'

# Your custom trained model path (set to None to use built-in model above)
FINETUNED_MODEL_PATH = None

# Output options
VISUALIZE = True
EXPORT_MODEL = True


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main training execution."""

    print("\n" + "=" * 70)
    print("STARDIST MODEL TRAINING")
    print("=" * 70)
    print(f"Model Name: {MODEL_NAME}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {MODEL_BASEDIR}")
    print("=" * 70)

    # Verify data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n Data directory not found: {DATA_DIR}")
        return

    # Setup GPU environment (Windows)
    setup_gpu_environment(CONDA_ENV_PATH)

    # Check GPU availability
    gpu_available = check_gpu()

    if not gpu_available:
        response = input("\nNo GPU detected. Training will be very slow. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # Run training pipeline
    try:
        model = run_training_pipeline(
            data_dir=DATA_DIR,
            model_name=MODEL_NAME,
            model_basedir=MODEL_BASEDIR,
            img_ext=IMG_EXT,
            min_nucleus_area=MIN_NUCLEUS_AREA,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            patch_size=PATCH_SIZE,
            use_augmentation=USE_AUGMENTATION,
            use_pretrained=USE_PRETRAINED,
            pretrained_model=PRETRAINED_MODEL,
            n_rays=N_RAYS,
            grid=GRID,
            copy_weights=COPY_WEIGHTS,
            visualize=VISUALIZE,
            export=EXPORT_MODEL
        )

        print("\n Training completed")
        print(f"   Model saved to: {os.path.join(MODEL_BASEDIR, MODEL_NAME)}")

    except Exception as e:
        print(f"\n Training failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        clear_gpu_memory()


if __name__ == '__main__':
    main()