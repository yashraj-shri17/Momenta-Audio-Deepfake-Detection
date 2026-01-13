
import os
import logging
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log")
        ]
    )

def convert_mp3_to_flac(mp3_file, output_folder="converted_flac"):
    """
    Converts an MP3 file to FLAC format.
    """
    if not os.path.exists(mp3_file):
        logger.error(f"File not found: {mp3_file}")
        raise FileNotFoundError(f"File not found: {mp3_file}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output folder: {output_folder}")
    
    filename = os.path.splitext(os.path.basename(mp3_file))[0]
    flac_file = os.path.join(output_folder, filename + ".flac")
    
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(flac_file, format="flac")
        logger.info(f"Converted {mp3_file} to {flac_file}")
        return flac_file
    except Exception as e:
        logger.error(f"Failed to convert {mp3_file} to FLAC: {e}")
        raise
