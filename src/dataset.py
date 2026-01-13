
import os
import torch
import torchaudio
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class ASVspoofDataset(Dataset):
    def __init__(self, dataset_path, protocol_file, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = []
        
        if not os.path.exists(protocol_file):
            logger.error(f"Protocol file not found: {protocol_file}")
            raise FileNotFoundError(f"Protocol file not found: {protocol_file}")
            
        # Parse the .train file
        skipped_count = 0
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                file_id, label = parts[1], parts[-2]  # Get file ID and label
                # In mini dataset, file_id might include relative path (e.g. "real/real_001")
                # But standard ASVspoof just has filename. 
                # Let's support both.
                
                if not file_id.endswith(".flac"):
                     file_path = f"{file_id}.flac"
                else:
                     file_path = file_id
                     
                audio_path = os.path.join(dataset_path, file_path)
                
                if os.path.exists(audio_path):
                    self.data.append((audio_path, label))
                else:
                    skipped_count += 1
        
        logger.info(f"Loaded {len(self.data)} samples. Skipped {skipped_count} missing files.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return a dummy tensor or handle appropriately. 
            # For simplicity in training loop, we might want to filter these out beforehand or raise error.
            # Here I'll raise to stop training on bad data.
            raise e
        
        if self.transform:
            waveform = self.transform(waveform)
        
        # Ensure waveform is the right length/shape if expecting fixed input size? 
        # The original code didn't handle variable lengths, but CNNs usually need fixed size or global pooling.
        # The original code just passed it through. I will keep it as is but add a comment.
        
        label_id = 1 if label == 'bonafide' else 0  # Convert labels to binary
        return waveform, torch.tensor(label_id, dtype=torch.float32)
