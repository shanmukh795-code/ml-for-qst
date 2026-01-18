import numpy as np
import torch
from torch.utils.data import Dataset

# Pauli Matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
PAULIS = [SIGMA_X, SIGMA_Y, SIGMA_Z]

def random_density_matrix():
    """Generates a random single-qubit density matrix (Haar random)."""
    # Generate a random pure state |psi>
    real_part = np.random.randn(2)
    imag_part = np.random.randn(2)
    psi = real_part + 1j * imag_part
    psi /= np.linalg.norm(psi)
    
    # Create density matrix rho = |psi><psi|
    rho = np.outer(psi, np.conj(psi))
    return rho

def get_pauli_probabilities(rho, pauli_idx):
    """Calculates probabilities of measuring +1 and -1 for a given Pauli."""
    pauli = PAULIS[pauli_idx]
    
    # Projectors for +1 and -1 eigenvalues
    # For single qubit Paulis, eigenvalues are always +1, -1.
    # We can find eigenvectors or use trace formula: P(+1) = Tr(rho * (I + P)/2)
    
    I = np.eye(2, dtype=np.complex64)
    proj_plus = (I + pauli) / 2
    
    prob_plus = np.real(np.trace(rho @ proj_plus))
    # Clip for numerical stability
    prob_plus = np.clip(prob_plus, 0.0, 1.0)
    
    return prob_plus

def simulate_measurement(rho, num_shots=100):
    """
    Simulates random Pauli measurements.
    Returns: List of (basis_index, outcome) tuples.
             Outcome is mapped to 0 (-1) and 1 (+1) or similar.
    """
    measurements = []
    
    for _ in range(num_shots):
        # Randomly choose a basis: 0=X, 1=Y, 2=Z
        basis = np.random.randint(0, 3)
        
        prob_plus = get_pauli_probabilities(rho, basis)
        
        # Sample outcome: +1 with prob_plus, -1 with 1-prob_plus
        if np.random.rand() < prob_plus:
            outcome = 1 # eigenvalue +1
        else:
            outcome = 0 # eigenvalue -1
            
        measurements.append((basis, outcome))
        
    return np.array(measurements)

class QuantumDataset(Dataset):
    def __init__(self, num_samples=1000, num_shots=100):
        self.num_samples = num_samples
        self.num_shots = num_shots
        self.data = []
        
        for _ in range(num_samples):
            rho = random_density_matrix()
            meas = simulate_measurement(rho, num_shots)
            
            # Target is the flat density matrix (real, imag parts) or L matrix parameters.
            # Here we just store the full rho to calculate loss later.
            rho_flat = np.concatenate([rho.real.flatten(), rho.imag.flatten()])
            
            self.data.append({
                'measurements': torch.tensor(meas, dtype=torch.long),
                'target_rho': torch.tensor(rho_flat, dtype=torch.float32)
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    # Test generation
    ds = QuantumDataset(num_samples=5, num_shots=10)
    print("Sample 0 measurements shape:", ds[0]['measurements'].shape)
    print("Sample 0 target shape:", ds[0]['target_rho'].shape)
