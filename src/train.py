import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from data import QuantumDataset
from model import DensityMatrixReconstructionModel
from utils import compute_fidelity, compute_trace_distance

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_SAMPLES = 5000
NUM_SHOTS = 100

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    dataset = QuantumDataset(num_samples=NUM_SAMPLES, num_shots=NUM_SHOTS)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Model
    model = DensityMatrixReconstructionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    print("Starting Training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            measurements = batch['measurements'].to(device) # (B, Shots, 2)
            target = batch['target_rho'].to(device)       # (B, 8)
            
            optimizer.zero_grad()
            output = model(measurements)
            
            # Loss = 1 - Mean Fidelity
            fidelity = compute_fidelity(output, target)
            loss = 1 - fidelity.mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_fidelity = []
        val_trace_dist = []
        
        with torch.no_grad():
            for batch in test_loader:
                measurements = batch['measurements'].to(device)
                target = batch['target_rho'].to(device)
                
                output = model(measurements)
                
                fid = compute_fidelity(output, target)
                td = compute_trace_distance(output, target)
                
                val_fidelity.extend(fid.cpu().numpy())
                val_trace_dist.extend(td.cpu().numpy())
        
        mean_fid = np.mean(val_fidelity)
        mean_td = np.mean(val_trace_dist)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Fidelity: {mean_fid:.4f} | Val TraceDist: {mean_td:.4f}")

    total_time = time.time() - start_time
    print(f"Training Complete. Total Time: {total_time:.2f}s")
    
    # Save Model
    save_path = "outputs/model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Latency Measurement
    measure_latency(model, device)

def measure_latency(model, device):
    dummy_input = torch.randint(0, 2, (1, NUM_SHOTS, 2)).to(device)
    model.eval()
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
        
    start = time.time()
    runs = 100
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end = time.time()
    
    latency = (end - start) / runs * 1000 # ms
    print(f"Inference Latency: {latency:.2f} ms per reconstruction where N_shots={NUM_SHOTS}")

if __name__ == "__main__":
    train()
