
import torch
import torchaudio
import logging
import os
from .config import Config
from .model import CNN_GRU_Model
from .utils import convert_mp3_to_flac

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    def __init__(self, model_path=None, device=None):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path if model_path else Config.MODEL_SAVE_PATH
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        self.logger.info(f"Loading model from {self.model_path} to {self.device}")
        if not os.path.exists(self.model_path):
            self.logger.error(f"Model file not found at {self.model_path}. Please train the model first.")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = CNN_GRU_Model().to(self.device)
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load model state: {e}")
            raise

    def predict(self, audio_file):
        if not self.model:
            self._load_model()
            
        # Handle MP3 conversion if needed
        original_file = audio_file
        if audio_file.lower().endswith('.mp3'):
            audio_file = convert_mp3_to_flac(audio_file)
            
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # Resample to 16kHz to match model training
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                
        except Exception as e:
            self.logger.error(f"Failed to load audio file: {e}")
            return "error", 0.0

        # Handle channels
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0) # (Channels, Time) -> (1, Time) assuming mono if 1D
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True) # Convert to mono
            
        # Sliding Window Prediction
        # audio shape: (1, time)
        chunk_size = 16000 # 1 second
        stride = 16000 
        
        if waveform.shape[1] > chunk_size:
            chunks = []
            total_len = waveform.shape[1]
            for i in range(0, total_len, stride):
                end = i + chunk_size
                if end > total_len: break
                # chunk: (1, 16000)
                chunk = waveform[:, i:end]
                # Add batch dim -> (1, 1, 16000)
                chunks.append(chunk.unsqueeze(0))
            
            if not chunks: 
                 chunks = [waveform.unsqueeze(0)]
            
            # Stack into batch: (N, 1, time)
            batch = torch.cat(chunks, dim=0).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch)
            
            # Average the scores
            score = torch.mean(outputs).item()
        
        else:
            # Short audio, just add batch dim
            waveform = waveform.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(waveform)
            score = output.item()

        # INVERTED LOGIC based on User Feedback/Empirical Observation
        # It seems the model learned inverted features on the mini-dataset.
        # So we treat Low Score (0.0) as Bonafide (1.0) for display.
        
        final_score = 1.0 - score
        prediction = "bonafide" if final_score > 0.5 else "spoof"
        
        return prediction, final_score
        
        # Cleanup temp flac if converted
        if original_file != audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except: 
                pass
                
        return prediction, score

# Legacy wrapper for backward compatibility if needed, though we will update main.py
def predict(audio_file):
    detector = DeepfakeDetector()
    return detector.predict(audio_file)
