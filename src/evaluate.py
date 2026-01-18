import torch
import numpy as np
import time
import os

from data import QuantumDataset
from model import DensityMatrixReconstructionModel
from utils import compute_fidelity, compute_trace_distance

MODEL_PATH = "outputs/model.pt"
NUM_SAMPLES = 1000
NUM_SHOTS = 100

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    model = DensityMatrixReconstructionModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    # Generate Test Data
    print(f"Generating {NUM_SAMPLES} test samples...")
    dataset = QuantumDataset(num_samples=NUM_SAMPLES, num_shots=NUM_SHOTS)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    fidelities = []
    trace_dists = []
    
    print("Running Evaluation...")
    with torch.no_grad():
        for batch in loader:
            measurements = batch['measurements'].to(device)
            target = batch['target_rho'].to(device)
            
            output = model(measurements)
            
            fid = compute_fidelity(output, target)
            td = compute_trace_distance(output, target)
            
            fidelities.extend(fid.cpu().numpy())
            trace_dists.extend(td.cpu().numpy())
            
    mean_fid = np.mean(fidelities)
    std_fid = np.std(fidelities)
    mean_td = np.mean(trace_dists)
    std_td = np.std(trace_dists)
    
    print("\n=== Final Metrics ===")
    print(f"Mean Fidelity:       {mean_fid:.4f} \u00b1 {std_fid:.4f}")
    print(f"Mean Trace Distance: {mean_td:.4f} \u00b1 {std_td:.4f}")
    
    # Latency Measurement
    print("\nMeasuring Inference Latency...")
    dummy_input = torch.randint(0, 2, (1, NUM_SHOTS, 2)).to(device)
    
    # Warmup
    for _ in range(50):
        _ = model(dummy_input)
        
    runs = 1000
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end = time.time()
    
    latency_ms = ((end - start) / runs) * 1000
    print(f"Inference Latency:   {latency_ms:.3f} ms/sample")

if __name__ == "__main__":
    evaluate()
