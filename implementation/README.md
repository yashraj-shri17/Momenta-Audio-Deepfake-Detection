
# Deepfake Detection Implementation

This directory contains the source code for the Momenta Audio Deepfake Detection model.

## Structure
- `main.py`: Entry point for the application (train/predict).
- `src/`: Source code package.
  - `config.py`: Configuration (paths, hyperparameters).
  - `dataset.py`: Dataset loading logic.
  - `model.py`: Neural network architecture.
  - `train.py`: Training loop.
  - `predict.py`: Inference logic.
  - `utils.py`: Utilities (logging, audio conversion).

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have `ffmpeg` installed and in your PATH for MP3 conversion support.*

2. Configure:
   Edit `src/config.py` or check the file to see which Environment Variables you can set to override defaults (e.g. `DATASET_PATH`).

## Usage

### Training
To train the model:
```bash
python main.py train
```
Ensure your dataset is correctly placed as per `src/config.py` Config.

### Prediction
To predict on a single audio file:
```bash
python main.py predict path/to/audio.mp3
```

## Troubleshooting
- If you encounter `OSError: [WinError 127]` with `torchaudio`, please reinstall torch and torchaudio with the correct version for your CUDA/CPU setup.
