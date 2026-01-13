
import os

class Config:
    # Data Paths
    # Data Paths
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Strictly use Mini Dataset
    _mini_root = os.path.join(_base_dir, "mini_dataset")
    if os.path.exists(_mini_root):
        PROTOCOL_FILE = os.getenv("PROTOCOL_FILE", os.path.join(_mini_root, "protocol.txt"))
        DATASET_PATH = os.getenv("DATASET_PATH", _mini_root)
    else:
        # Production / Inference Mode
        PROTOCOL_FILE = None
        DATASET_PATH = None

    MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "models/cnn_gru_model.pth")
    
    # Training Hyperparameters
    # Training Hyperparameters
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 2)) 
    EPOCHS = int(os.getenv("EPOCHS", 10))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
    
    # Device
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Audio Processing
    SAMPLE_RATE = 16000 # Standard for many models
    
    @staticmethod
    def ensure_directories():
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
