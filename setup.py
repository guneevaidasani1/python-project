import os
import zipfile
import sys
from config import DATASET_CONFIG, DATA_DIR

def unzip_datasets():
    """
    Checks if the data directory exists and contains the unzipped files.
    If not, it attempts to unzip the corresponding zip files into DATA_DIR.
    """
    
    required_zips = [
        "UCMerced_LandUse.zip",
        "Medical_Waste_4_0.zip",
        "Fetus_US.zip"
    ]
    
    print("="*60)
    print("       DATASET SETUP: Checking for unzipped data files...      ")
    print("="*60)
    print("🚨 CRITICAL STEP: Please ensure the following ZIP files are in the project root are named exactly as mentioned:")
    for name in required_zips:
        print(f"   - {name}")
    print("-" * 60)

    all_unzipped = True
    
    
    datasets_to_check = {k: v for k, v in DATASET_CONFIG.items() if isinstance(v, dict)}
    
    for key, config in datasets_to_check.items():
        local_name = config['local_name']
        zip_file = config.get('zip_file_name')
        
        
        if config.get('path_suffix'):
            expected_path = os.path.join(DATA_DIR, local_name, config['path_suffix'])
        else:
            expected_path = os.path.join(DATA_DIR, local_name)

        if os.path.exists(expected_path) and os.path.isdir(expected_path):
            print(f"✅ Data for '{local_name}' found at: {expected_path}")
            continue

        all_unzipped = False
        
        if not zip_file:
            print(f"❌ ERROR: Zip file name not defined for {local_name} in config.py.")
            sys.exit(1)
            
        zip_path = os.path.join(".", zip_file) 
        
        if os.path.exists(zip_path):
            print(f"📦 Unzipping '{zip_file}' to '{DATA_DIR}'...")
            try:
                
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(DATA_DIR)
                print(f"🎉 Success! '{zip_file}' extracted.")
            except Exception as e:
                print(f"💥 FATAL ERROR: Could not unzip {zip_file}. Ensure the zip structure is correct.")
                print(f"Error details: {e}")
                sys.exit(1)
        else:
            print(f"⚠️ WARNING: Expected directory '{expected_path}' not found.")
            print(f"    AND required zip file '{zip_file}' not found at '{zip_path}'.")
            print("    Please ensure you hav run the unzip code/script.")
            sys.exit(1)

    if all_unzipped:
        print("\n✅ All necessary datasets are ready.")
        
if __name__ == '__main__':
    unzip_datasets()