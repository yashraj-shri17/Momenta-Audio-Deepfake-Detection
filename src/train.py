
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from .config import Config
from .model import CNN_GRU_Model
from .dataset import ASVspoofDataset

logger = logging.getLogger(__name__)

def train():
    logger.info("Starting training process...")
    
    if not os.path.exists(Config.PROTOCOL_FILE) or not os.path.exists(Config.DATASET_PATH):
        logger.error("Dataset details missing. Please check configuration.")
        return

    train_dataset = ASVspoofDataset(Config.DATASET_PATH, Config.PROTOCOL_FILE)
    if len(train_dataset) == 0:
        logger.error("No data found in dataset. Exiting.")
        return

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")
    
    model = CNN_GRU_Model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    model.train()
    
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0
        
        for batch in train_loader:
            inputs, labels = batch
            # Ensure proper input shape (Batch, 1, Length)
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Accuracy calc
            predicted = (outputs.view(-1) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0
        logger.info(f"Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save the model
    Config.ensure_directories()
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    logger.info(f"Model saved to {Config.MODEL_SAVE_PATH}")
