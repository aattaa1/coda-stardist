# Import pipeline functions
from segmentation.segmentation_functions_depr import *
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Set environment variable before importing TensorFlow
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
CONDA_ENV_PATH = r"C:\Users\aattaa1\AppData\Local\miniconda3\envs\coda-stardist" # GPU Environment (Windows only - set to your conda env path)
cuda_path = [
    os.path.join(CONDA_ENV_PATH, 'Library', 'bin')]

for cuda_path in cuda_path:
    if os.path.exists(cuda_path):
        os.environ['PATH'] = cuda_path + os.pathsep + os.environ.get('PATH', '')
        print(f"Added to PATH: {cuda_path}")

def process_image_list(model_path, model_name, wsi_dir, save_geojson, save_mat, save_pkl, save_geojson_every):
    """Main pipeline execution."""

    if not os.path.exists(wsi_dir):
        print(f"\ Directory not found: {wsi_dir}")
        return

    print("\n" + "=" * 70)
    print("STARDIST WSI SEGMENTATION")
    print("=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)

    # Setup GPU environment (Windows)
    setup_gpu_environment(CONDA_ENV_PATH)

    # Check GPU availability
    gpu_available = check_gpu()
    if not gpu_available:
        response = input("\n No GPU detected. Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # Load model
    print("\n" + "-" * 50)
    print("Loading StarDist model...")
    print("-" * 50)

    # Process directory
    process_image_directory(
        wsi_dir=wsi_dir,
        model_file=os.path.join(model_path, f'{model_name}'),
        output_base=os.path.join(wsi_dir, f'{model_name}'), # output folder
        wsi_extensions=('.ndpi', '.svs', '.scn'), # wsi file extensions
        tile_extensions=('.tif', '.tiff', '.png', '.jpg', '.jpeg'), # tile file extensions
        make_mat_file=save_mat, # save mat file
        make_pkl_file=save_pkl, # save pkl file
        generate_geojson=save_geojson, # whether to save geojson file
        geojson_every_n=save_geojson_every,
        extract_rgb=True, # get mean and std intensity from images
        ds_amt=1.0 , # downsampling (1 = 20x, 2 = 10x)
        classification_color=[97, 214, 59], # make the masks green in qupath
        inverse_order=False, # invert order of selection
        block_size=4096, # tile size for inference
        min_overlap=128, # tile overlap
        context=128,
        n_tiles=(8, 8, 1) # tiles to analyze at once
    )

    # Cleanup
    clear_gpu_memory()

if __name__ == '__main__':

    # model information
    model_path = r"\\10.17.182.53\kiemen-lab-data\Ali Attaa\nuclear segmentation\stardist models"
    model_name = "stardist_WCC_02_19_2026"

    # list of folders containing images to segment
    wsi_list = [
        r''
        # Add more directories as needed
    ]

    # output file options
    save_geojson, save_mat, save_pkl = True, True, False  # filetypes to save
    save_geojson_every = 50  # Generate GeoJSON every N images (1 = every image)

    for wsi_dir in wsi_list:
        process_image_list(model_path, model_name, wsi_dir, save_geojson, save_mat, save_pkl, save_geojson_every)