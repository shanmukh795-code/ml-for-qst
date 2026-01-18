import torch
import numpy as np

def compute_fidelity(rho_pred, rho_target):
    """
    Computes Fidelity between two batches of density matrices.
    F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2
    For pure states matches |<psi|phi>|^2.
    
    Inputs are (B, 8) flattened real/imag parts.
    """
    # Reshape to (B, 2, 2) complex
    # Format: [r00, r01_r, r10_r, r11, i00, i01_i, i10_i, i11] 
    # Wait, my model output format was:
    # real_flat = [n_rho00, n_rho01_real, n_rho01_real, n_rho11]
    # imag_flat = [0, n_rho01_imag, -n_rho01_imag, 0]
    
    # Let's reconstruct consistent complex matrices
    def reshape_rho(flat):
        # flat: (B, 8) -> real (B, 4), imag (B, 4)
        real = flat[:, :4].view(-1, 2, 2)
        imag = flat[:, 4:].view(-1, 2, 2)
        return torch.complex(real, imag)

    rho_p = reshape_rho(rho_pred)
    rho_t = reshape_rho(rho_target)
    
    # We operate on CPU with numpy/scipy for matrix square roots if not available in PyTorch for batches easily.
    # PyTorch has matrix_exp but sqrt is tricky for batches of complex matrices sometimes.
    # However, since matrices are 2x2, we can do it analytically or use linalg.
    
    # fidelity = (Tr sqrt(rho_p^1/2 rho_t rho_p^1/2))^2
    # For general mixed states, this is expensive.
    # Since rho_t is rank-1 (pure state from data.py), 
    # Fidelity reduces to <psi| rho_p |psi>.
    # Wait, data.py generates pure states rho = |psi><psi|.
    # so F = <psi| rho_p |psi> = Tr(rho_t * rho_p)
    
    # Let's check if target is pure. Yes, in data.py it is pure.
    # So F = Tr(rho_pred @ rho_target)
    
    # calculation
    prod = torch.matmul(rho_p, rho_t)
    f = torch.real(torch.diagonal(prod, dim1=-2, dim2=-1).sum(-1))
    
    # Clamp to [0, 1]
    f = torch.clamp(f, 0.0, 1.0)
    return f

def compute_trace_distance(rho_pred, rho_target):
    """
    Computes Trace Distance: 0.5 * Tr|rho_pred - rho_target|
    """
    def reshape_rho(flat):
        real = flat[:, :4].view(-1, 2, 2)
        imag = flat[:, 4:].view(-1, 2, 2)
        return torch.complex(real, imag)
        
    rho_p = reshape_rho(rho_pred)
    rho_t = reshape_rho(rho_target)
    
    diff = rho_p - rho_t
    
    # Eigenvalues of difference
    # For 2x2, we can compute eigenvalues easily
    evals = torch.linalg.eigvalsh(diff) 
    # Trace norm is sum of absolute eigenvalues
    trace_norm = torch.abs(evals).sum(dim=-1)
    
    return 0.5 * trace_norm
