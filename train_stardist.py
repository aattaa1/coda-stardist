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

# Set environment variable before importing TensorFlow
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import training functions


# ============================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================

# GPU Environment (Windows only - set to your conda env path)
CONDA_ENV_PATH = r"C:\Users\YOUR_USERNAME\miniconda3\envs\stardist_pipeline"

# Training data directory
# Expected structure:
#   DATA_DIR/
#       image1.tif
#       image1.geojson
#       image2.tif
#       image2.geojson
#       ...
DATA_DIR = r"Z:\Your\Training\Data\Directory"

# Model output settings
MODEL_NAME = "My_Custom_Model"
MODEL_BASEDIR = r"Z:\Your\Models\Directory"

# Data settings
IMG_EXT = ".tif"  # Image file extension
MIN_NUCLEUS_AREA = 20  # Minimum nucleus area (pixels) to include
VALIDATION_SPLIT = 0.2  # Fraction of data for validation

# Training settings
EPOCHS = 400  # Number of training epochs
STEPS_PER_EPOCH = 200  # Steps per epoch
PATCH_SIZE = (256, 256)  # Training patch size
USE_AUGMENTATION = True  # Use data augmentation

# Model initialization
USE_PRETRAINED = True  # Start from pretrained weights (recommended)
PRETRAINED_MODEL = "HCC_HE_Finetuned"  # Pretrained model to use

# Output options
VISUALIZE = True  # Show/save visualizations
EXPORT_MODEL = True  # Export for TensorFlow Serving


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
        print(f"\n❌ Data directory not found: {DATA_DIR}")
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
            visualize=VISUALIZE,
            export=EXPORT_MODEL
        )

        print("\n✅ Training completed successfully!")
        print(f"   Model saved to: {os.path.join(MODEL_BASEDIR, MODEL_NAME)}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        clear_gpu_memory()


if __name__ == '__main__':
    main()