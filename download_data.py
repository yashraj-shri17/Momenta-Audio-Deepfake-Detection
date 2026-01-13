
import os
import requests
import zipfile
import tqdm
from src.utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

DATA_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
DEST_FOLDER = "dataset"
ZIP_FILE = os.path.join(DEST_FOLDER, "LA.zip")

def download_file(url, filename):
    if os.path.exists(filename):
        logger.info(f"File {filename} already exists. Skipping download.")
        return

    logger.info(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm.tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    logger.info("Download complete.")

def extract_file(zip_path, extract_to):
    logger.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info("Extraction complete.")

def main():
    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)
    
    try:
        download_file(DATA_URL, ZIP_FILE)
        extract_file(ZIP_FILE, DEST_FOLDER)
        
        # Update config hint
        extracted_path = os.path.join(DEST_FOLDER, "LA")
        protocol_path = os.path.join(extracted_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
        audio_path = os.path.join(extracted_path, "ASVspoof2019_LA_train", "flac")
        
        print("\n" + "="*50)
        print("Dataset Ready!")
        print(f"Protocol File: {protocol_path}")
        print(f"Audio Path: {audio_path}")
        print("="*50)
        print("\nPlease update your src/config.py with these paths.")
        
    except Exception as e:
        logger.error(f"Failed to setup dataset: {e}")

if __name__ == "__main__":
    main()
