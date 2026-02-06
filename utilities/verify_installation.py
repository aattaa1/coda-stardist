"""
StarDist Pipeline - Installation Verification Script
=====================================================
Run this script to verify all packages are installed correctly.

Usage:
    python verify_installation.py

Author: Ali Attaa
Date: 02/05/2026
"""

import sys
import os


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_status(name: str, status: bool, version: str = None, details: str = None) -> None:
    """Print status of a check."""
    if status:
        if version:
            print(f"  ✅ {name}: {version}")
        else:
            print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}: FAILED")

    if details:
        print(f"      {details}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"  ⚠️  {message}")


def check_python_version() -> bool:
    """Check Python version."""
    print_header("1. Python Version")

    version = sys.version.split()[0]
    major, minor = int(version.split('.')[0]), int(version.split('.')[1])

    is_valid = (major == 3 and minor == 10)
    print_status(f"Python {version}", is_valid)

    if not is_valid:
        if major == 3 and minor > 10:
            print_warning("Python 3.10.x recommended for TensorFlow 2.10 compatibility")
        elif major == 3 and minor < 10:
            print_warning("Python version too old. Please use Python 3.10.x")
        else:
            print_warning("Python 3.x required")

    return is_valid


def check_tensorflow() -> bool:
    """Check TensorFlow installation and GPU availability."""
    print_header("2. TensorFlow & GPU")

    all_passed = True

    # Check TensorFlow import
    try:
        import tensorflow as tf
        print_status("TensorFlow", True, tf.__version__)

        # Check if correct version
        if not tf.__version__.startswith("2.10"):
            print_warning(f"TensorFlow 2.10.x recommended, found {tf.__version__}")

        # Check CUDA build
        cuda_built = tf.test.is_built_with_cuda()
        print_status("Built with CUDA", cuda_built)

        if not cuda_built:
            print_warning("TensorFlow not built with CUDA - GPU acceleration unavailable")

        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            print_status(f"GPU(s) detected", True, f"{len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"      GPU {i}: {gpu.name}")

            # Try to enable memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print_status("GPU memory growth", True, "enabled")
            except RuntimeError as e:
                print_status("GPU memory growth", False)
                print_warning(f"Could not enable memory growth: {e}")
        else:
            print_status("GPU(s) detected", False)
            print_warning("No GPU found - pipeline will use CPU (slower)")
            all_passed = False

    except ImportError as e:
        print_status("TensorFlow", False)
        print_warning(f"Import error: {e}")
        all_passed = False
    except Exception as e:
        print_status("TensorFlow", False)
        print_warning(f"Error: {e}")
        all_passed = False

    return all_passed


def check_stardist() -> bool:
    """Check StarDist installation."""
    print_header("3. StarDist & CSBDeep")

    all_passed = True

    # Check StarDist
    try:
        import stardist
        print_status("StarDist", True, stardist.__version__)

        # Check StarDist2D model class
        try:
            from stardist.models import StarDist2D, Config2D
            print_status("StarDist2D model class", True)
        except ImportError as e:
            print_status("StarDist2D model class", False)
            print_warning(f"Import error: {e}")
            all_passed = False

        # Check predict_instances_big
        try:
            from stardist.models import StarDist2D
            if hasattr(StarDist2D, 'predict_instances_big'):
                print_status("predict_instances_big", True, "available")
            else:
                print_status("predict_instances_big", False)
                print_warning("Method not found - update StarDist")
                all_passed = False
        except Exception as e:
            print_status("predict_instances_big", False)
            all_passed = False

    except ImportError as e:
        print_status("StarDist", False)
        print_warning(f"Import error: {e}")
        all_passed = False

    # Check CSBDeep
    try:
        import csbdeep
        print_status("CSBDeep", True, csbdeep.__version__)

        # Check normalize function
        try:
            from csbdeep.utils import normalize
            print_status("csbdeep.utils.normalize", True)
        except ImportError:
            print_status("csbdeep.utils.normalize", False)
            all_passed = False

    except ImportError as e:
        print_status("CSBDeep", False)
        print_warning(f"Import error: {e}")
        all_passed = False

    return all_passed


def check_image_processing() -> bool:
    """Check image processing packages."""
    print_header("4. Image Processing")

    all_passed = True

    packages = [
        ("numpy", "numpy", None),
        ("pandas", "pandas", None),
        ("scipy", "scipy", None),
        ("scikit-image", "skimage", None),
        ("OpenCV", "cv2", None),
        ("Pillow", "PIL", "PIL.Image"),
        ("tifffile", "tifffile", None),
        ("imagecodecs", "imagecodecs", None),
    ]

    for name, module, submodule in packages:
        try:
            if submodule:
                exec(f"from {module} import {submodule.split('.')[-1]}")
                mod = __import__(module)
            else:
                mod = __import__(module)

            version = getattr(mod, "__version__", "OK")
            print_status(name, True, version)

        except ImportError as e:
            print_status(name, False)
            print_warning(f"Import error: {e}")
            all_passed = False
        except Exception as e:
            print_status(name, False)
            print_warning(f"Error: {e}")
            all_passed = False

    return all_passed


def check_openslide() -> bool:
    """Check OpenSlide installation."""
    print_header("5. OpenSlide (WSI Support)")

    all_passed = True

    # Check openslide-python
    try:
        import openslide
        print_status("openslide-python", True, openslide.__version__)

        # Check OpenSlide library version
        try:
            lib_version = openslide.__library_version__
            print_status("OpenSlide library", True, lib_version)
        except AttributeError:
            # Older versions may not have this attribute
            print_status("OpenSlide library", True, "version unknown")

        # Check if OpenSlide can be used
        try:
            from openslide import OpenSlide
            print_status("OpenSlide class", True)
        except ImportError as e:
            print_status("OpenSlide class", False)
            print_warning(f"Import error: {e}")
            all_passed = False

    except ImportError as e:
        print_status("openslide-python", False)
        print_warning(f"Import error: {e}")
        print_warning("Install with: pip install openslide-bin openslide-python")
        all_passed = False
    except OSError as e:
        print_status("OpenSlide library", False)
        print_warning(f"Library error: {e}")
        print_warning("OpenSlide DLLs may not be in PATH")
        all_passed = False
    except Exception as e:
        print_status("OpenSlide", False)
        print_warning(f"Error: {e}")
        all_passed = False

    return all_passed


def check_geojson_geometry() -> bool:
    """Check GeoJSON and geometry packages."""
    print_header("6. GeoJSON & Geometry")

    all_passed = True

    # Check geojson
    try:
        import geojson
        version = getattr(geojson, '__version__', 'OK')
        print_status("geojson", True, version)
    except ImportError as e:
        print_status("geojson", False)
        print_warning(f"Import error: {e}")
        all_passed = False

    # Check shapely
    try:
        import shapely
        print_status("shapely", True, shapely.__version__)

        # Check specific imports
        try:
            from shapely.geometry import shape, Polygon
            print_status("shapely.geometry", True)
        except ImportError:
            print_status("shapely.geometry", False)
            all_passed = False

    except ImportError as e:
        print_status("shapely", False)
        print_warning(f"Import error: {e}")
        all_passed = False

    return all_passed


def check_visualization() -> bool:
    """Check visualization and utility packages."""
    print_header("7. Visualization & Utilities")

    all_passed = True

    packages = [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
    ]

    for name, module in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "OK")
            print_status(name, True, version)
        except ImportError as e:
            print_status(name, False)
            print_warning(f"Import error: {e}")
            all_passed = False

    return all_passed


def check_file_io() -> bool:
    """Check file I/O capabilities."""
    print_header("8. File I/O")

    all_passed = True

    # Check scipy.io for MAT files
    try:
        from scipy.io import savemat, loadmat
        print_status("scipy.io (MAT files)", True)
    except ImportError as e:
        print_status("scipy.io (MAT files)", False)
        print_warning(f"Import error: {e}")
        all_passed = False

    # Check pickle
    try:
        import pickle
        print_status("pickle", True)
    except ImportError as e:
        print_status("pickle", False)
        print_warning(f"Import error: {e}")
        all_passed = False

    # Check json
    try:
        import json
        print_status("json", True)
    except ImportError as e:
        print_status("json", False)
        print_warning(f"Import error: {e}")
        all_passed = False

    # Check h5py for HDF5 files
    try:
        import h5py
        print_status("h5py (HDF5 files)", True, h5py.__version__)
    except ImportError as e:
        print_status("h5py (HDF5 files)", False)
        print_warning(f"Import error: {e}")
        all_passed = False

    return all_passed


def check_training_dependencies() -> bool:
    """Check training-specific dependencies."""
    print_header("9. Training Dependencies")

    all_passed = True

    # Check augmend
    try:
        import augmend
        version = getattr(augmend, '__version__', 'OK')
        print_status("augmend (augmentation)", True, version)
    except ImportError as e:
        print_status("augmend (augmentation)", False)
        print_warning(f"Import error: {e}")
        print_warning("Required for training. Install with: pip install augmend")
        all_passed = False

    # Check imageio
    try:
        import imageio
        print_status("imageio", True, imageio.__version__)
    except ImportError as e:
        print_status("imageio", False)
        print_warning(f"Import error: {e}")
        all_passed = False

    return all_passed


def run_import_test() -> bool:
    """Test importing the main pipeline modules."""
    print_header("10. Pipeline Module Import Test")

    all_passed = True

    # Test segmentation functions import
    try:
        # Check if file exists
        seg_path = os.path.join(os.path.dirname(__file__), '..', 'segmentation', 'segmentation_functions.py')
        if os.path.exists(seg_path):
            print_status("segmentation/segmentation_functions.py", True, "found")
        else:
            # Try current directory
            seg_path = os.path.join(os.path.dirname(__file__), 'segmentation_functions.py')
            if os.path.exists(seg_path):
                print_status("segmentation_functions.py", True, "found")
            else:
                print_status("segmentation_functions.py", False)
                print_warning("File not found in expected locations")
    except Exception as e:
        print_status("segmentation module", False)
        print_warning(f"Error: {e}")
        all_passed = False

    # Test training functions import
    try:
        train_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'training_functions.py')
        if os.path.exists(train_path):
            print_status("training/training_functions.py", True, "found")
        else:
            train_path = os.path.join(os.path.dirname(__file__), 'training_functions.py')
            if os.path.exists(train_path):
                print_status("training_functions.py", True, "found")
            else:
                print_status("training_functions.py", False)
                print_warning("File not found in expected locations")
    except Exception as e:
        print_status("training module", False)
        print_warning(f"Error: {e}")
        all_passed = False

    return all_passed


def print_summary(results: dict) -> bool:
    """Print summary of all checks."""
    print_header("VERIFICATION SUMMARY")

    total_checks = len(results)
    passed_checks = sum(results.values())
    failed_checks = total_checks - passed_checks

    print(f"\n  Total checks: {total_checks}")
    print(f"  Passed: {passed_checks}")
    print(f"  Failed: {failed_checks}")

    all_passed = (failed_checks == 0)

    if all_passed:
        print("""
  ✅ ALL CHECKS PASSED!

  Your environment is ready to run the StarDist pipeline.

  Next steps:
    1. Edit segmentation/apply_stardist.py with your paths
    2. Run: python segmentation/apply_stardist.py from CLI or run direclty from IDE
        """)
    else:
        print("""
  ⚠️  SOME CHECKS FAILED

  Please review the errors above and install missing packages.

  Common fixes:
    - GPU not detected: Check CUDA path and driver installation
    - Package missing: pip install <package_name>
    - OpenSlide error: pip install openslide-bin openslide-python

  For detailed installation instructions, see INSTALL.md
        """)

    return all_passed


def main() -> bool:
    """Run all verification checks."""
    print("=" * 60)
    print("STARDIST PIPELINE - INSTALLATION VERIFICATION")
    print("=" * 60)

    results = {}

    # Run all checks
    results['Python'] = check_python_version()
    results['TensorFlow'] = check_tensorflow()
    results['StarDist'] = check_stardist()
    results['Image Processing'] = check_image_processing()
    results['OpenSlide'] = check_openslide()
    results['GeoJSON'] = check_geojson_geometry()
    results['Visualization'] = check_visualization()
    results['File I/O'] = check_file_io()
    results['Training'] = check_training_dependencies()
    results['Pipeline Modules'] = run_import_test()

    # Print summary
    all_passed = print_summary(results)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)