
import torch
import torch.nn as nn

class CNN_GRU_Model(nn.Module):
    def __init__(self):
        super(CNN_GRU_Model, self).__init__()
        # Input expected: (Batch, Channels, Length)
        # Note: Audio length must be sufficient for convolutions.
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, 1, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Permute for GRU: (batch, features, seq_len) -> (batch, seq_len, features)
        x = x.permute(0, 2, 1)  
        
        x, _ = self.gru(x)
        
        # Take the output of the last time step
        # x shape from GRU: (batch, seq_len, hidden_size)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)
