"""
StarDist WSI Segmentation Pipeline - Functions Module
=====================================================
Contains all functions for segmentation, feature extraction, and file I/O.

Author: Ali Attaa
Date: 02/04/2026
"""

import os
import gc
import cv2
import json
import time
import pickle
import geojson
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from tifffile import imread, imwrite
from scipy.io import savemat, loadmat
from openslide import OpenSlide
from skimage.transform import rescale
from stardist.models import StarDist2D, Config2D

Image.MAX_IMAGE_PIXELS = None


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
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️ Could not set memory growth: {e}")
    else:
        print("❌ No GPU found - using CPU")

    print("=" * 60)
    return len(gpus) > 0


def clear_gpu_memory() -> None:
    """Clear TensorFlow session and run garbage collection."""
    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================
# FILE UTILITIES
# ============================================================

def get_sorted_files(directory: str, *extensions, filter_str: str = None) -> List[str]:
    """
    Get sorted list of files matching specified extensions.

    Args:
        directory: Directory to search
        *extensions: File extensions to match (e.g., '.ndpi', '.svs')
        filter_str: Optional string to filter filenames

    Returns:
        Sorted list of file paths
    """
    if not os.path.exists(directory):
        return []
    return sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith(extensions) and (filter_str in f if filter_str else True)
    ])


def extract_slide_number(filename: str) -> int:
    """
    Extract numeric slide identifier from filename.

    Args:
        filename: Filename to parse

    Returns:
        Extracted number or 0 if none found
    """
    import re
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0


def create_output_directories(base_path: str, subdirs: List[str]) -> Dict[str, str]:
    """
    Create output directory structure.

    Args:
        base_path: Base output directory
        subdirs: List of subdirectory names to create

    Returns:
        Dictionary mapping subdir names to full paths
    """
    paths = {'base': base_path}
    os.makedirs(base_path, exist_ok=True)

    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)
        paths[subdir] = path

    return paths


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_path: str) -> StarDist2D:
    """
    Load StarDist model with custom weights, config, and thresholds.

    Args:
        model_path: Path to model directory containing config.json,
                   thresholds.json, and weights_best.h5

    Returns:
        Loaded StarDist2D model
    """
    print(f"Loading model from: {model_path}")

    config_path = os.path.join(model_path, 'config.json')
    thresh_path = os.path.join(model_path, 'thresholds.json')
    weights_path = os.path.join(model_path, 'weights_best.h5')

    # Verify files exist
    for path, name in [(config_path, 'config.json'),
                       (thresh_path, 'thresholds.json'),
                       (weights_path, 'weights_best.h5')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    with open(thresh_path, 'r') as f:
        thresh = json.load(f)

    model = StarDist2D(config=Config2D(**config), basedir=model_path, name='model')
    model.thresholds = thresh
    print(f"  Thresholds: {model.thresholds}")
    model.load_weights(weights_path)
    print("  ✅ Model loaded successfully!")

    return model


# ============================================================
# PIXEL RESOLUTION
# ============================================================

def extract_and_save_pixel_sizes(wsi_dir: str, output_dir: str,
                                 wsi_extensions: Tuple[str, ...] = ('.ndpi', '.svs')) -> None:
    """
    Extract pixel sizes from WSI files and save as .mat files.

    Args:
        wsi_dir: Directory containing WSI files
        output_dir: Directory to save .mat files
        wsi_extensions: Tuple of valid WSI extensions
    """
    os.makedirs(output_dir, exist_ok=True)

    wsi_files = [f for f in os.listdir(wsi_dir) if f.endswith(wsi_extensions)]

    for filename in sorted(wsi_files):
        mat_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.mat")

        if os.path.exists(mat_path):
            continue

        wsi_path = os.path.join(wsi_dir, filename)
        print(f"  Extracting pixel resolution: {filename}")

        try:
            slide = OpenSlide(wsi_path)
            pix_res = {
                'x': slide.properties.get('openslide.mpp-x', '0'),
                'y': slide.properties.get('openslide.mpp-y', '0')
            }
            slide.close()
            savemat(mat_path, {'pix_res': pix_res})
        except Exception as e:
            print(f"    ⚠️ Could not extract pixel resolution: {e}")


# ============================================================
# SEGMENTATION
# ============================================================

def segment_image(img_path: str, model: StarDist2D,
                  block_size: int = 4096,
                  min_overlap: int = 128,
                  context: int = 128,
                  n_tiles: Tuple[int, int, int] = (4, 4, 1)) -> Tuple[np.ndarray, Dict]:
    """
    Segment image using appropriate method based on size.

    Args:
        img_path: Path to image file
        model: Loaded StarDist model
        block_size: Block size for predict_instances_big
        min_overlap: Minimum overlap between blocks
        context: Context size for blocks
        n_tiles: Number of tiles for prediction

    Returns:
        Tuple of (mask, polys dictionary)
    """
    print(f"  Reading image...")
    img = imread(img_path)
    img = img / 255.0  # Normalize

    print(f"  Image shape: {img.shape}")

    height, width = img.shape[:2]
    total_pixels = height * width

    # Use regular predict_instances for small images
    if total_pixels <= 4_000_000:  # ~2000x2000 or smaller
        print(f"  Running segmentation (small image mode)...")
        mask, polys = model.predict_instances(img)
    else:
        # Use predict_instances_big for large images
        print(f"  Running segmentation (block_size={block_size})...")
        try:
            mask, polys = model.predict_instances_big(
                img, axes='YXC',
                block_size=block_size,
                min_overlap=min_overlap,
                context=context,
                n_tiles=n_tiles
            )
        except Exception:
            print(f"  Retrying with block_size={block_size // 2}...")
            try:
                mask, polys = model.predict_instances_big(
                    img, axes='YXC',
                    block_size=block_size // 2,
                    min_overlap=min_overlap,
                    context=context,
                    n_tiles=n_tiles
                )
            except Exception:
                print(f"  Falling back to standard segmentation...")
                mask, polys = model.predict_instances(img)

    print(f"  Detected {len(polys['points'])} nuclei")

    del img
    gc.collect()

    return mask, polys


# ============================================================
# GEOJSON OUTPUT
# ============================================================

def save_geojson_from_polys(polys: Dict, output_path: str, name: str,
                            ds_amt: float = 1.0,
                            classification_name: str = 'Nuclei',
                            classification_color: List[int] = [97, 214, 59]) -> str:
    """
    Save GeoJSON directly from StarDist prediction output.

    Args:
        polys: Dictionary with 'coord' and 'points' from prediction
        output_path: Directory to save GeoJSON
        name: Base filename
        ds_amt: Downsampling amount
        classification_name: Classification name for QuPath
        classification_color: RGB color for QuPath

    Returns:
        Path to saved GeoJSON file
    """
    coords = polys['coord']
    points = polys['points']

    geo_data = []

    for label_id, (point, contour) in enumerate(zip(points, coords), start=1):
        # Convert contour to list format and apply downsampling
        contour_list = [[float(xy[0]) / ds_amt, float(xy[1]) / ds_amt] for xy in contour.T]

        # Swap coordinates (y, x) -> (x, y) for GeoJSON and close polygon
        contour_swapped = [[c[1], c[0]] for c in contour_list]
        contour_swapped.append(contour_swapped[0])  # Close polygon

        geo_data.append({
            "type": "Feature",
            "id": "PathCellObject",
            "geometry": {
                "type": "Polygon",
                "coordinates": [contour_swapped]
            },
            "properties": {
                'objectType': 'annotation',
                'label': label_id,
                'name': str(label_id),
                'classification': {
                    'name': classification_name,
                    'color': classification_color
                }
            }
        })

    geojson_path = os.path.join(output_path, f"{name}.geojson")
    with open(geojson_path, 'w') as f:
        geojson.dump(geo_data, f)

    print(f"  Saved GeoJSON: {name}.geojson ({len(geo_data)} nuclei)")
    return geojson_path


# ============================================================
# FEATURE EXTRACTION UTILITIES
# ============================================================

def cntarea(cnt: np.ndarray) -> float:
    """Calculate contour area using OpenCV."""
    return cv2.contourArea(cnt.astype(np.float32))


def cntperi(cnt: np.ndarray) -> float:
    """Calculate contour perimeter using OpenCV."""
    return cv2.arcLength(cnt.astype(np.float32), True)


def cntMA(cnt: np.ndarray) -> List[float]:
    """
    Calculate major axis, minor axis, and orientation from contour.

    Args:
        cnt: Contour points as numpy array

    Returns:
        List of [major_axis, minor_axis, orientation]
    """
    cnt = cnt.astype(np.float32)
    if len(cnt) < 5:
        return [1.0, 1.0, 0.0]
    try:
        (x, y), (MA, ma), orientation = cv2.fitEllipse(cnt)
        return [max(MA, ma), min(MA, ma), orientation]
    except:
        return [1.0, 1.0, 0.0]


def get_rgb_avg_std(centroid: List[int], contour: np.ndarray,
                    offset: int, wsi_image: np.ndarray) -> Tuple[float, ...]:
    """
    Extract RGB mean and std from region around centroid.

    Args:
        centroid: [y, x] centroid coordinates
        contour: Contour points as numpy array
        offset: Radius for cropping region
        wsi_image: Full image array

    Returns:
        Tuple of (r_mean, g_mean, b_mean, r_std, g_std, b_std)
    """
    # Centroid is [y, x], need to handle appropriately
    cy, cx = centroid[0], centroid[1]

    x_low = max(0, cx - offset)
    x_high = min(wsi_image.shape[1], cx + offset)
    y_low = max(0, cy - offset)
    y_high = min(wsi_image.shape[0], cy + offset)

    # Adjust contour to local coordinates
    local_contour = contour.copy()
    local_contour[:, 0] -= x_low  # x
    local_contour[:, 1] -= y_low  # y

    cropped = wsi_image[y_low:y_high, x_low:x_high]

    if cropped.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Create mask
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_contour.astype(np.int32)], 1)

    # Extract pixels
    pixels = cropped[mask == 1]

    if len(pixels) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    r_mean, g_mean, b_mean = pixels.mean(axis=0)
    r_std, g_std, b_std = pixels.std(axis=0)

    return float(r_mean), float(g_mean), float(b_mean), float(r_std), float(g_std), float(b_std)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features_from_polys(polys: Dict, wsi_path: str,
                                extract_rgb: bool = True,
                                rgb_offset: int = 30) -> pd.DataFrame:
    """
    Extract nuclear features directly from StarDist prediction output.

    Args:
        polys: Dictionary with 'coord' and 'points' from prediction
        wsi_path: Path to WSI for RGB extraction
        extract_rgb: Whether to extract RGB features
        rgb_offset: Radius for RGB extraction region

    Returns:
        DataFrame with nuclear features
    """
    coords = polys['coord']
    points = polys['points']

    if len(points) == 0:
        return pd.DataFrame()

    nm = os.path.splitext(os.path.basename(wsi_path))[0]

    # Load WSI for RGB extraction if needed
    wsi_image = None
    if extract_rgb:
        print(f"    Loading WSI for RGB extraction...")
        wsi_image = imread(wsi_path)

    # Initialize feature lists
    centroids_x, centroids_y = [], []
    areas, perimeters, circularities, aspect_ratios = [], [], [], []
    compactness_a, eccentricity_a, extent_a, form_factor_a = [], [], [], []
    maximum_radius_a, mean_radius_a, median_radius_a = [], [], []
    minor_axis_length_a, major_axis_length_a, orientation_degrees_a = [], [], []
    solidity_a, convex_area_a, equiv_diameter_a = [], [], []
    r_avg_list, g_avg_list, b_avg_list = [], [], []
    r_std_list, g_std_list, b_std_list = [], [], []

    for j in tqdm(range(len(points)), desc="    Extracting features", leave=False):
        point = points[j]
        contour_raw = coords[j]

        # Centroid (StarDist returns [y, x])
        centroid = [int(point[0]), int(point[1])]
        centroids_x.append(centroid[1])  # x
        centroids_y.append(centroid[0])  # y

        # Convert contour for OpenCV (needs [x, y] format as int32)
        contour = np.array([[int(xy[1]), int(xy[0])] for xy in contour_raw.T], dtype=np.int32)

        # Shape features
        area = cntarea(contour)
        perimeter = cntperi(contour)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        MA, ma, orientation = cntMA(contour)
        aspect_ratio = MA / ma if ma > 0 else 1

        compactness = (perimeter ** 2) / area if area > 0 else 0
        eccentricity = np.sqrt(1 - (ma / MA) ** 2) if MA > ma and MA > 0 else 0
        extent = area / (MA * ma) if MA * ma > 0 else 0
        form_factor = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0

        convex_area =cv2.contourArea(cv2.convexHull(contour)) if len(contour) > 3 else area
        solidity = area / convex_area if convex_area > 0 else 0
        equiv_diameter = np.sqrt(4 *area / np.pi) if area > 0 else 0

        # Radius features
        centroid_arr = np.array([centroid[1], centroid[0]])  # [x, y]
        distances = np.linalg.norm(contour - centroid_arr, axis=1) if len(contour) > 0 else [0]

        # Append shape features
        areas.append(area)
        perimeters.append(perimeter)
        circularities.append(circularity)
        aspect_ratios.append(aspect_ratio)
        compactness_a.append(compactness)
        eccentricity_a.append(eccentricity)
        extent_a.append(extent)
        form_factor_a.append(form_factor)
        maximum_radius_a.append(np.max(distances))
        mean_radius_a.append(np.mean(distances))
        median_radius_a.append(np.median(distances))
        minor_axis_length_a.append(ma)
        major_axis_length_a.append(MA)
        orientation_degrees_a.append(np.degrees(orientation))
        solidity_a.append(solidity)
        convex_area_a.append(convex_area)
        equiv_diameter_a.append(equiv_diameter)

        # RGB features
        if extract_rgb and wsi_image is not None:
            r_avg, g_avg, b_avg, r_std, g_std, b_std = get_rgb_avg_std(
                centroid, contour, rgb_offset, wsi_image
            )
            r_avg_list.append(r_avg)
            g_avg_list.append(g_avg)
            b_avg_list.append(b_avg)
            r_std_list.append(r_std)
            g_std_list.append(g_std)
            b_std_list.append(b_std)

    # Build DataFrame
    dat = {
        'Centroid_x': centroids_x,
        'Centroid_y': centroids_y,
        'Area': areas,
        'Perimeter': perimeters,
        'Circularity': circularities,
        'Aspect_Ratio': aspect_ratios,
        'compactness': compactness_a,
        'eccentricity': eccentricity_a,
        'extent': extent_a,
        'form_factor': form_factor_a,
        'maximum_radius': maximum_radius_a,
        'mean_radius': mean_radius_a,
        'median_radius': median_radius_a,
        'minor_axis_length': minor_axis_length_a,
        'major_axis_length': major_axis_length_a,
        'orientation_degrees': orientation_degrees_a,
        'solidity': solidity_a,
        'convex_area': convex_area_a,
        'equiv_diameter': equiv_diameter_a,
        'slide_num': [extract_slide_number(nm)] * len(points)
    }

    if extract_rgb and wsi_image is not None:
        dat.update({
            'r_mean_intensity': r_avg_list,
            'g_mean_intensity': g_avg_list,
            'b_mean_intensity': b_avg_list,
            'r_std': r_std_list,
            'g_std': g_std_list,
            'b_std': b_std_list
        })

    # Cleanup
    if wsi_image is not None:
        del wsi_image
        gc.collect()

    return pd.DataFrame(dat).astype(np.float32)


# ============================================================
# MAT FILE OUTPUT
# ============================================================

def save_features_to_mat(df: pd.DataFrame, output_path: str, name: str,
                         pixres_dir: str = None) -> str:
    """
    Save features DataFrame to MAT file with pixel resolution.

    Args:
        df: Features DataFrame
        output_path: Directory to save MAT file
        name: Base filename
        pixres_dir: Directory containing pixel resolution MAT files

    Returns:
        Path to saved MAT file
    """
    # Load pixel resolution if available
    pix_res = {}
    if pixres_dir:
        pixres_path = os.path.join(pixres_dir, f"{name}.mat")
        if os.path.exists(pixres_path):
            pixres_data = loadmat(pixres_path)
            pix_res = pixres_data.get('pix_res', {})

    mat_data = {
        'features': df.to_numpy().astype(np.float32),
        'feature_names': df.columns.tolist(),
        'pix_res': pix_res,
        'num_nuclei': len(df)
    }

    mat_path = os.path.join(output_path, f"{name}.mat")
    savemat(mat_path, mat_data)

    return mat_path


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_single_image(wsi_path: str, model: StarDist2D,
                         out_geojson: str, out_pkl: str, out_mat: str,
                         pixres_dir: str,
                         idx: int = 1, total: int = 1,
                         generate_geojson: bool = True,
                         geojson_every_n: int = 1,
                         extract_rgb: bool = True,
                         ds_amt: float = 1.0,
                         classification_name: str = 'Nuclei',
                         classification_color: List[int] = [97, 214, 59],
                         block_size: int = 4096,
                         min_overlap: int = 128,
                         context: int = 128,
                         n_tiles: Tuple[int, int, int] = (4, 4, 1)) -> bool:
    """
    Process a single image: segment, extract features, save outputs.

    Args:
        wsi_path: Path to image file
        model: Loaded StarDist model
        out_geojson: Output directory for GeoJSON
        out_pkl: Output directory for pickle files
        out_mat: Output directory for MAT files
        pixres_dir: Directory with pixel resolution files
        idx: Current index
        total: Total number of files
        generate_geojson: Whether to generate GeoJSON files
        geojson_every_n: Generate GeoJSON every N images
        extract_rgb: Whether to extract RGB features
        ds_amt: Downsampling amount for GeoJSON
        classification_name: Classification name for QuPath
        classification_color: RGB color for QuPath
        block_size: Block size for predict_instances_big
        min_overlap: Minimum overlap between blocks
        context: Context size for blocks
        n_tiles: Number of tiles for prediction

    Returns:
        True if successful, False otherwise
    """
    name = os.path.basename(wsi_path)
    nm = os.path.splitext(name)[0]

    # Check if already processed
    pkl_path = os.path.join(out_pkl, f"{nm}.pkl")
    if os.path.exists(pkl_path):
        print(f"Skipping {nm} ({idx}/{total}) - already processed")
        return True

    print(f"\n{'=' * 60}")
    print(f"Processing: {nm} ({idx}/{total})")
    print(f"{'=' * 60}")

    start_time = time.time()
    success = False

    try:
        # Step 1: Segment
        mask, polys = segment_image(
            wsi_path, model,
            block_size=block_size,
            min_overlap=min_overlap,
            context=context,
            n_tiles=n_tiles
        )

        if len(polys['points']) == 0:
            print(f"  ⚠️ No nuclei detected in {nm}")
            return True

        # Step 2: Save GeoJSON (if enabled and on correct interval)
        if generate_geojson and (idx % geojson_every_n == 0):
            print(f"  Saving GeoJSON...")
            save_geojson_from_polys(
                polys, out_geojson, nm,
                ds_amt=ds_amt,
                classification_name=classification_name,
                classification_color=classification_color
            )

        # Step 3: Extract features
        print(f"  Extracting features...")
        df = extract_features_from_polys(polys, wsi_path, extract_rgb)

        if len(df) > 0:
            # Save pickle
            df.to_pickle(pkl_path)
            print(f"  Saved: {nm}.pkl ({len(df)} nuclei)")

            # Save MAT file
            mat_path = save_features_to_mat(df, out_mat, nm, pixres_dir)
            print(f"  Saved: {nm}.mat")

        elapsed = (time.time() - start_time) / 60
        print(f"  ✅ Completed in {elapsed:.1f} minutes")
        success = True

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    clear_gpu_memory()

    return success


# ============================================================
# BATCH PROCESSING
# ============================================================

def process_image_directory(wsi_dir: str, model: StarDist2D,
                            output_base: str,
                            wsi_extensions: Tuple[str, ...] = ('.ndpi', '.svs'),
                            tile_extensions: Tuple[str, ...] = ('.tif', '.tiff', '.png', '.jpg', '.jpeg'),
                            generate_geojson: bool = True,
                            geojson_every_n: int = 50,
                            extract_rgb: bool = True,
                            ds_amt: float = 1.0,
                            classification_name: str = 'Nuclei',
                            classification_color: List[int] = [97, 214, 59],
                            inverse_order: bool = False,
                            block_size: int = 4096,
                            min_overlap: int = 128,
                            context: int = 128,
                            n_tiles: Tuple[int, int, int] = (4, 4, 1)) -> Dict:
    """
    Process all images in a directory.

    Args:
        wsi_dir: Directory containing images
        model: Loaded StarDist model
        output_base: Base output directory
        wsi_extensions: Tuple of WSI file extensions
        tile_extensions: Tuple of tile file extensions
        generate_geojson: Whether to generate GeoJSON files
        geojson_every_n: Generate GeoJSON every N images
        extract_rgb: Whether to extract RGB features
        ds_amt: Downsampling amount for GeoJSON
        classification_name: Classification name for QuPath
        classification_color: RGB color for QuPath
        inverse_order: Process files in reverse order
        block_size: Block size for predict_instances_big
        min_overlap: Minimum overlap between blocks
        context: Context size for blocks
        n_tiles: Number of tiles for prediction

    Returns:
        Dictionary with processing statistics
    """
    print(f"\n{'=' * 70}")
    print(f"Processing directory: {wsi_dir}")
    print(f"{'=' * 70}")

    # Setup output directories
    out_geojson = os.path.join(output_base, 'geojsons')
    out_pkl = os.path.join(output_base, 'feature_pickles')
    out_mat = os.path.join(output_base, 'feature_mat')
    out_pixres = os.path.join(wsi_dir, 'segmentation_analysis', 'pix_res_info')

    for d in [output_base, out_geojson, out_pkl, out_mat, out_pixres]:
        os.makedirs(d, exist_ok=True)

    print(f"Output directory: {output_base}")

    # Get image files
    wsi_files = get_sorted_files(wsi_dir, *wsi_extensions)
    if len(wsi_files) == 0:
        wsi_files = get_sorted_files(wsi_dir, *tile_extensions)

    print(f"Found {len(wsi_files)} image files")

    if len(wsi_files) == 0:
        print("No image files found!")
        return {'total': 0, 'processed': 0, 'skipped': 0, 'failed': 0}

    # Extract pixel resolutions
    print("\n" + "-" * 50)
    print("Extracting pixel resolutions...")
    print("-" * 50)
    extract_and_save_pixel_sizes(wsi_dir, out_pixres, wsi_extensions)

    # Process images
    print("\n" + "-" * 50)
    print("Processing images...")
    print("-" * 50)

    total = len(wsi_files)
    processed = 0
    skipped = 0
    failed = 0

    if inverse_order:
        wsi_files = list(reversed(wsi_files))
        indices = range(total, 0, -1)
    else:
        indices = range(1, total + 1)

    for idx, wsi_path in zip(indices, wsi_files):
        nm = os.path.splitext(os.path.basename(wsi_path))[0]
        pkl_path = os.path.join(out_pkl, f"{nm}.pkl")

        if os.path.exists(pkl_path):
            skipped += 1
            print(f"Skipping {nm} ({idx}/{total}) - already processed")
            continue

        success = process_single_image(
            wsi_path, model,
            out_geojson, out_pkl, out_mat, out_pixres,
            idx=idx, total=total,
            generate_geojson=generate_geojson,
            geojson_every_n=geojson_every_n,
            extract_rgb=extract_rgb,
            ds_amt=ds_amt,
            classification_name=classification_name,
            classification_color=classification_color,
            block_size=block_size,
            min_overlap=min_overlap,
            context=context,
            n_tiles=n_tiles
        )

        if success:
            processed += 1
        else:
            failed += 1

    stats = {
        'total': total,
        'processed': processed,
        'skipped': skipped,
        'failed': failed
    }

    print(f"\n✅ Completed: {wsi_dir}")
    print(f"   Total: {total}, Processed: {processed}, Skipped: {skipped}, Failed: {failed}")

    return stats