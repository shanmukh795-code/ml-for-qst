import torch
import torch.nn as nn
import numpy as np

class MeasurementEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        # 3 bases (X, Y, Z) * 2 outcomes (-1, 1) = 6 possible tokens
        self.embedding = nn.Embedding(6, embed_dim)

    def forward(self, basis, outcome):
        # Basis: 0, 1, 2
        # Outcome: 0, 1
        # Token ID = basis * 2 + outcome -> 0..5
        # 0: X-, 1: X+, 2: Y-, 3: Y+, 4: Z-, 5: Z+
        token_ids = basis * 2 + outcome
        return self.embedding(token_ids)

class DensityMatrixReconstructionModel(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, num_layers=2):
        super().__init__()
        self.embedding = MeasurementEmbedding(embed_dim)
        
        # Transformer Encoder to aggregate information from all shots
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output Head to predict Cholesky factor L (2x2 lower triangular)
        # L has 4 parameters: L00(real), L11(real), L10_real, L10_imag
        # We also need to predict these from the aggregated representation.
        # We can use the [CLS] token concept or mean pooling.
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4) # 4 params for single qubit L
        )

    def forward(self, measurements):
        # measurements: (B, N_shots, 2) -> last dim is (basis, outcome)
        basis = measurements[:, :, 0]
        outcome = measurements[:, :, 1]
        
        x = self.embedding(basis, outcome) # (B, N_shots, D)
        
        # Transformer encoding
        x = self.encoder(x) # (B, N_shots, D)
        
        # Mean pooling over shots to get fixed size vector
        x = x.mean(dim=1) # (B, D)
        
        # Predict parameters for L
        params = self.output_head(x) # (B, 4)
        
        # Construct L
        # L = [[L00, 0], [L10_r + i L10_i, L11]]
        # Diagonal elements L00, L11 must be real.
        # To ensure unique Cholesky, we often force diagonals > 0, 
        # but for reconstruction just L L^dag suffices, diagonals can be arbitrary real.
        # We will predict raw values.
        
        L00 = params[:, 0]
        L11 = params[:, 1]
        L10_real = params[:, 2]
        L10_imag = params[:, 3]
        
        # Construct rho = L L^dag
        # We do this manually for batch support
        
        # First row L[0,:] = [L00, 0]
        # Second row L[1,:] = [L10, L11] where L10 = r + i*im
        
        # rho00 = |L00|^2
        rho00 = L00**2
        
        # rho11 = |L10|^2 + |L11|^2
        rho11 = (L10_real**2 + L10_imag**2) + L11**2
        
        # rho01 = L00 * conj(L10) = L00 * (r - i*im)
        rho01_real = L00 * L10_real
        rho01_imag = L00 * (-L10_imag)
        
        # rho10 = conj(rho01)
        
        # Form Batch Rho (B, 2, 2) (Complex)
        # We return Real and Imag parts separately (B, 2, 2, 2) or just keep tensors
        
        # Flattened trace for normalization
        trace = rho00 + rho11
        
        # Normalize
        # Add epsilon to avoid div by zero
        trace = trace.unsqueeze(1).unsqueeze(2) + 1e-8
        
        # Reconstruct normalized rho elements
        n_rho00 = rho00 / trace.squeeze()
        n_rho11 = rho11 / trace.squeeze()
        
        n_rho01_real = rho01_real / trace.squeeze()
        n_rho01_imag = rho01_imag / trace.squeeze()
        
        # Stack to (B, 4) -> [r00, r01_r, r01_i, r11] (Ignoring r10 as it's redundant)
        # Or return full matrix components
        
        # Let's return the full density matrix (real, imag) parts flattened: 
        # [rho00_r, rho00_i, rho01_r, rho01_i, rho10_r, rho10_i, rho11_r, rho11_i]
        # imaginary parts of diag are 0.
        
        zeros = torch.zeros_like(n_rho00)
        
        # Flat output: Real parts then Imag parts? Or interleaved?
        # Let's match the data.py format: real flatten, imag flatten.
        # rho = [[rho00, rho01], [rho10, rho11]]
        
        real_flat = torch.stack([n_rho00, n_rho01_real, n_rho01_real, n_rho11], dim=1) # Note: rho10_real = rho01_real
        imag_flat = torch.stack([zeros, n_rho01_imag, -n_rho01_imag, zeros], dim=1)   # rho10_imag = -rho01_imag
        
        # Output shape (B, 8)
        output = torch.cat([real_flat, imag_flat], dim=1)
        
        return output

if __name__ == "__main__":
    model = DensityMatrixReconstructionModel()
    dummy_input = torch.randint(0, 2, (3, 100, 2)) # (B, Shots, 2)
    output = model(dummy_input)
    print("Model Output Shape:", output.shape) # Should be (3, 8)
    print("Trace of first sample:", output[0, 0] + output[0, 3]) # Should be 1.0
