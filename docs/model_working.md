# Part 1: Model Working

## 1. Project Overview
For this assignment, I chose **Track 1** to solve the Quantum State Tomography (QST) problem. The goal is to figure out the state of a single qubit (its density matrix $\rho$) just by looking at a bunch of measurement outcomes, which are called "Classical Shadows". 

I decided to use a **Transformer-based** neural network for this because the measurement data comes in as a sequence of shots, and Transformers are really good at handling sequences where the order doesn't strictly matter but the collection of data points gives the full picture.

## 2. How the Model Works

### 2.1 The Input (Classical Shadows)
The input to my model is a list of $N$ "shots". For each shot, we measure the qubit in one of the Pauli bases ($X, Y,$ or $Z$) and get a result ($+1$ or $-1$).
To feed this into the neural network, I used an **Embedding Layer**.
- The model takes pairs of `(Basis, Outcome)`.
- It converts these pairs into dense vectors so the network can learn from them.

### 2.2 The Architecture (Transformer Encoder)
I used a **Transformer Encoder** as the backbone of my model.
- **Why?** Since the "shadows" are just a collection of independent measurements, the specific order (shot 1 vs shot 100) doesn't really change the physical state. The Self-Attention mechanism in the Transformer allows the model to look at all the shots together and figure out which correlations matter.
- After processing the sequence, I take the **average (mean pooling)** of all the output vectors to get a single summary vector that represents the quantum state.

### 2.3 Enforcing Physical Constraints
This was the most important part of the assignment. A valid density matrix $\rho$ has to satisfy three strict rules:
1.  **Hermitian**: It must be equal to its conjugate transpose.
2.  **Positive Semi-Definite (PSD)**: All eigenvalues must be non-negative.
3.  **Unit Trace**: The diagonal elements must sum to 1.

If I just let the neural network output a random $2 \times 2$ matrix, it would almost certainly break these rules. So, instead of predicting $\rho$ directly, I predicted a lower triangular matrix $L$ and used the **Cholesky Decomposition** trick:
$$ \rho_{\text{unnorm}} = L L^\dagger $$
By construction, any matrix written as $L L^\dagger$ is automatically Hermitian and PSD.

To fix the trace, I simply normalized it at the end:
$$ \rho = \frac{\rho_{\text{unnorm}}}{\text{Tr}(\rho_{\text{unnorm}})} $$

This way, no matter what the neural network outputs, the result is *always* a valid quantum state.

## 3. Training Strategy
I trained the model using **Quantum Fidelity** as the loss function. Fidelity basically measures how "close" two quantum states are.
- **Loss**: $1 - \text{Fidelity}$
- I minimized this loss so that the Fidelity would get as close to 1.0 as possible.

Overall, this approach ensures that we combine the learning power of deep learning with the strict laws of quantum mechanics.
