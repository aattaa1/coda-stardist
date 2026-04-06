"""
StarDist WSI Segmentation Pipeline - Main Execution Script
==========================================================
Run this script to process WSIs or image tiles.

Usage:
    python apply_stardist.py

Author: Ali Attaa
Date: 02/04/2026
"""
# Import functions
from segmentation.segmentation_functions import *
import os
import time

# Set environment variable before importing TensorFlow
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================

# GPU Environment (Windows only - set to your conda env path)
CONDA_ENV_PATH = r"C:\Users\aattaa1\AppData\Local\miniconda3\envs\coda-stardist"

# Input: List of directories containing WSIs or image tiles
WSI_PATHS = [
    r'\\10.99.134.183\kiemen-lab-data\Ali Attaa\Ali Liver\HE Liver Blocks\S22-35490_2D'
    # Add more directories as needed
]

# Model settings
MODEL_PATH = r"\\10.99.134.183\kiemen-lab-data\Ali Attaa\nuclear segmentation\stardist models"
MODEL_NAME = "stardist_WCC_02_19_2026"


# Output date stamp (auto-generated)
DATE = time.strftime("%m_%d_%Y")

# Processing options
EXTRACT_RGB_FEATURES, GENERATE_GEOJSON, SAVE_PKL, SAVE_MAT, INVERSE_ORDER = True, True, True, True, False
GEOJSON_EVERY_N = 50


# Segmentation parameters
BLOCK_SIZE = 4096  # Block size for large images (reduce if out of memory)
MIN_OVERLAP = 128
CONTEXT = 128
N_TILES = (4, 4, 1)

# GeoJSON settings
DS_AMT = 1.0  # Downsampling (1 = 20x, 2 = 10x)
CLASSIFICATION_NAME = "Nuclei"
CLASSIFICATION_COLOR = [97, 214, 59]  # Green

# File extensions
WSI_EXTENSIONS = ('.ndpi', '.svs')
TILE_EXTENSIONS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')


# ============================================================
# MAIN EXECUTION (DO NOT CHANGE)
# ============================================================

def main():
    """Main pipeline execution."""

    print("\n" + "=" * 70)
    print("STARDIST WSI SEGMENTATION")
    print("=" * 70)
    print(f"Date: {DATE}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    # Setup GPU environment (Windows)
    setup_gpu_environment(CONDA_ENV_PATH)

    # Check GPU availability
    gpu_available = check_gpu()

    if not gpu_available:
        response = input("\nNo GPU detected. Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # Load model
    print("\n" + "-" * 50)
    print("Loading StarDist model...")
    print("-" * 50)

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f" Failed to load model: {e}")
        return

    # Process each directory
    all_stats = []

    for wsi_dir in WSI_PATHS:
        if not os.path.exists(wsi_dir):
            print(f"\n Directory not found: {wsi_dir}")
            continue

        # Create output directory name
        output_base = os.path.join(wsi_dir, f'StarDist_{MODEL_NAME}')

        # Process directory
        stats = process_image_directory(
            wsi_dir=wsi_dir,
            model_file=model,
            output_base=output_base,
            wsi_extensions=WSI_EXTENSIONS,
            tile_extensions=TILE_EXTENSIONS,
            generate_geojson=GENERATE_GEOJSON,
            geojson_every_n=GEOJSON_EVERY_N,
            make_pkl_file = SAVE_PKL,
            make_mat_file= SAVE_MAT,
            extract_rgb=EXTRACT_RGB_FEATURES,
            ds_amt=DS_AMT,
            classification_name=CLASSIFICATION_NAME,
            classification_color=CLASSIFICATION_COLOR,
            inverse_order=INVERSE_ORDER,
            block_size=BLOCK_SIZE,
            min_overlap=MIN_OVERLAP,
            context=CONTEXT,
            n_tiles=N_TILES
        )

        all_stats.append({'directory': wsi_dir, **stats})

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)

    total_processed = 0
    total_failed = 0

    for stat in all_stats:
        print(f"\n {stat['directory']}")
        print(f"   Total: {stat['total']}, Processed: {stat['processed']}, "
              f"Skipped: {stat['skipped']}, Failed: {stat['failed']}")
        total_processed += stat['processed']
        total_failed += stat['failed']

    print("\n" + "-" * 50)
    print(f"Total processed: {total_processed}")
    print(f"Total failed: {total_failed}")
    print("=" * 70)

    # Cleanup
    clear_gpu_memory()


if __name__ == '__main__':
    main()