"""
Dataset Download and Setup Script
Automates downloading and preparing ReID datasets
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def print_info(text):
    """Print info message"""
    print(f"[INFO] {text}")

def print_success(text):
    """Print success message"""
    print(f"[OK] {text}")

def print_error(text):
    """Print error message"""
    print(f"[ERROR] {text}")

def download_casia_b(output_dir):
    """Download and prepare CASIA-B gait dataset"""
    print_header("CASIA-B Gait Dataset Setup")

    casia_dir = output_dir / "CASIA-B"
    casia_pkl_dir = output_dir / "CASIA-B-pkl"

    print_info("CASIA-B requires a signed dataset agreement")
    print_info("Please follow these steps:")
    print("  1. Visit: http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp")
    print("  2. Download and sign the dataset release agreement")
    print("  3. Email the signed agreement to the dataset maintainers")
    print("  4. Wait for approval (typically 1-3 days)")
    print("  5. Download from: http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip")
    print()

    # Check if already downloaded
    if casia_dir.exists():
        print_success(f"CASIA-B directory already exists at {casia_dir}")

        # Offer to preprocess
        if not casia_pkl_dir.exists():
            print_info("Preprocessing CASIA-B for OpenGait...")
            try:
                pretreat_script = Path(__file__).parent / "OpenGait" / "datasets" / "pretreatment.py"
                if pretreat_script.exists():
                    subprocess.run([
                        sys.executable,
                        str(pretreat_script),
                        "--input_path", str(casia_dir),
                        "--output_path", str(casia_pkl_dir)
                    ], check=True)
                    print_success(f"Preprocessed dataset saved to {casia_pkl_dir}")
                else:
                    print_error(f"Pretreatment script not found at {pretreat_script}")
            except subprocess.CalledProcessError as e:
                print_error(f"Preprocessing failed: {e}")
        else:
            print_success(f"Preprocessed CASIA-B already exists at {casia_pkl_dir}")
    else:
        print_info(f"Once downloaded, extract to: {casia_dir}")
        print_info("Then run this script again to preprocess")

def download_market1501_kaggle(output_dir):
    """Download Market-1501 from Kaggle"""
    print_header("Market-1501 Dataset Download (Kaggle)")

    market_dir = output_dir / "Market-1501"

    # Check if already exists
    if market_dir.exists() and any(market_dir.iterdir()):
        print_success(f"Market-1501 already exists at {market_dir}")
        return

    print_info("Attempting to download Market-1501 from Kaggle...")
    print_info("Note: Requires Kaggle API credentials")
    print()

    # Check for Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print_error("Kaggle credentials not found")
        print("\nTo set up Kaggle API:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token'")
        print("  4. This downloads kaggle.json")
        print("  5. Place it in: " + str(Path.home() / ".kaggle" / "kaggle.json"))
        print()
        print("Alternative: Manual download from https://www.kaggle.com/datasets/pengcw1/market-1501")
        return

    # Install kaggle package if needed
    try:
        import kaggle
    except ImportError:
        print_info("Installing Kaggle package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        import kaggle

    # Download dataset
    try:
        print_info("Downloading Market-1501... (this may take a while)")
        market_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "pengcw1/market-1501",
            "-p", str(market_dir)
        ], check=True)

        # Unzip
        zip_file = market_dir / "market-1501.zip"
        if zip_file.exists():
            print_info("Extracting dataset...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(market_dir)
            zip_file.unlink()  # Remove zip after extraction
            print_success(f"Market-1501 downloaded and extracted to {market_dir}")
        else:
            print_error("Download completed but zip file not found")

    except subprocess.CalledProcessError as e:
        print_error(f"Kaggle download failed: {e}")
        print_info("Try manual download from: https://www.kaggle.com/datasets/pengcw1/market-1501")

def setup_directories(base_dir):
    """Create dataset directory structure"""
    print_header("Setting Up Dataset Directories")

    # Appearance-based datasets directory
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    print_success(f"Created: {datasets_dir}")

    # OpenGait datasets directory
    opengait_dir = base_dir / "OpenGait" / "datasets"
    opengait_dir.mkdir(parents=True, exist_ok=True)
    print_success(f"Created: {opengait_dir}")

    # Face database directory
    faces_dir = base_dir / "face-reidentification" / "assets" / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    print_success(f"Created: {faces_dir}")

    return datasets_dir, opengait_dir

def create_sample_faces(faces_dir):
    """Create README for custom faces"""
    readme = faces_dir / "README.txt"
    if not readme.exists():
        readme.write_text("""
Place face images here for person identification.

Filename format: person_name.jpg
Examples:
  - john_doe.jpg
  - jane_smith.jpg
  - bob_jones.jpg

The system will use the filename as the person's identity.

Supported formats: .jpg, .jpeg, .png
Recommended: Clear front-facing photos, 256x256 or larger
""")
        print_success(f"Created faces directory guide at {readme}")

def main():
    parser = argparse.ArgumentParser(description='Download and prepare ReID datasets')
    parser.add_argument('--dataset', choices=['market1501', 'casia-b', 'all'],
                       default='all', help='Dataset to download')
    parser.add_argument('--skip-download', action='store_true',
                       help='Only setup directories, skip downloads')

    args = parser.parse_args()

    base_dir = Path(__file__).parent
    print_header("Person ReID Dataset Setup")
    print_info(f"Base directory: {base_dir}")

    # Setup directories
    datasets_dir, opengait_dir = setup_directories(base_dir)
    faces_dir = base_dir / "face-reidentification" / "assets" / "faces"
    create_sample_faces(faces_dir)

    if args.skip_download:
        print_info("Skipping downloads (--skip-download specified)")
        print_success("Directory setup complete!")
        return

    # Download datasets
    if args.dataset in ['market1501', 'all']:
        download_market1501_kaggle(datasets_dir)

    if args.dataset in ['casia-b', 'all']:
        download_casia_b(opengait_dir)

    # Summary
    print_header("Setup Summary")
    print("Dataset locations:")
    print(f"  - Appearance datasets: {datasets_dir}")
    print(f"  - Gait datasets: {opengait_dir}")
    print(f"  - Face database: {faces_dir}")
    print()
    print("Next steps:")
    print("  1. Check DATASETS.md for more dataset options")
    print("  2. Add custom faces to face-reidentification/assets/faces/")
    print("  3. Train models or use pretrained models for inference")
    print()
    print_success("Setup complete!")

if __name__ == "__main__":
    main()
