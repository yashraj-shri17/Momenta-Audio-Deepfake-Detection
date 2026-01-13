import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment

# --------------------
# 1. Conversion from MP3 to FLAC
# --------------------
def convert_mp3_to_flac(mp3_file, output_folder="converted_flac"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    flac_file = os.path.join(output_folder, os.path.splitext(os.path.basename(mp3_file))[0] + ".flac")
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(flac_file, format="flac")
    return flac_file

# --------------------
# 2. Load ASVspoof Dataset
# --------------------
class ASVspoofDataset(Dataset):
    def __init__(self, dataset_path, protocol_file, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        
        # Parse the .train file
        self.data = []
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                file_id, label = parts[1], parts[-2]  # Get file ID and label
                audio_path = os.path.join(dataset_path, f"{file_id}.flac")
                if os.path.exists(audio_path):
                    self.data.append((audio_path, label))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        label = 1 if label == 'bonafide' else 0  # Convert labels to binary
        return waveform, label

# --------------------
# 3. Define CNN-GRU Model
# --------------------
class CNN_GRU_Model(nn.Module):
    def __init__(self):
        super(CNN_GRU_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Reshape for GRU
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

# --------------------
# 4. Train Model
# --------------------
def train_model(train_loader, model, criterion, optimizer, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    return model

# --------------------
# 5. Predict on New Audio
# --------------------
def predict_audio(model, mp3_file):
    flac_file = convert_mp3_to_flac(mp3_file)
    waveform, sample_rate = torchaudio.load(flac_file)
    waveform = waveform.unsqueeze(0)  # Add batch dimension
    
    model.eval()
    with torch.no_grad():
        output = model(waveform)
    
    prediction = "bonafide" if output.item() > 0.5 else "spoof"
    print(f"Prediction: {prediction}")
    return prediction

# --------------------
# 6. Run Training & Testing
# --------------------
if __name__ == "__main__":
    # Updated paths to point to mock_data for demonstration/testing
    base_dir = os.path.dirname(os.path.abspath(__file__))
    protocol_file = os.path.join(base_dir, "mock_data", "protocol.txt")
    dataset_path = os.path.join(base_dir, "mock_data", "audio")
    
    # Check if data exists
    if not os.path.exists(protocol_file) or not os.path.exists(dataset_path):
        print("Error: Mock data not found. Please ensure mock_data directory exists.")
        exit(1)

    train_dataset = ASVspoofDataset(dataset_path, protocol_file)
    # create DataLoader with drop_last if needed, but 2 samples is fine for 16 batch? No, it will just be size 2.
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    model = CNN_GRU_Model()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training on mock data...")
    trained_model = train_model(train_loader, model, criterion, optimizer, epochs=2)
    print("Training complete.")
    
    # Test with a generated FLAC file (since we don't have MP3 for test easily without ignoring conversion)
    # We will reuse one of the mock files for prediction test
    test_audio_file = os.path.join(dataset_path, "file_001.flac")
    print(f"Testing prediction on {test_audio_file}")
    
    # Bypass mp3 conversion for this test or handle it
    # predict_audio function expects MP3 and converts. We should make it flexible.
    # But for minimal changes to legacy code:
    def predict_audio_flexible(model, audio_file):
        if audio_file.endswith('.mp3'):
             audio_file = convert_mp3_to_flac(audio_file)
        # Load directly
        waveform, sample_rate = torchaudio.load(audio_file)
        if waveform.dim() == 1: waveform = waveform.unsqueeze(0)
        waveform = waveform.unsqueeze(0)  # Add batch dimension
        
        model.eval()
        with torch.no_grad():
            output = model(waveform)
        
        prediction = "bonafide" if output.item() > 0.5 else "spoof"
        print(f"Prediction: {prediction}")
        return prediction

    predict_audio_flexible(trained_model, test_audio_file)
