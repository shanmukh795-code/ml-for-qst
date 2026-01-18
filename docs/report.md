# Final Report: Single Qubit Density Matrix Reconstruction

## 1. Methodology (Track 1)
For this project, I implemented a **Transformer-based neural network** to solve the reconstruction problem. The core idea was to treat the measurement outcomes (Classical Shadows) as a sequence and use Self-Attention to aggregate the information.

To ensure strict physical validity, I implemented a custom output layer that constructs the density matrix using the Cholesky limit ($ \rho = LL^\dagger $). This guarantees that my output is always Hermitian and Positive Semi-Definite.

## 2. Results
I evaluated my model on a test set of 1000 random pure states. Here is how it performed:

| Metric | Result | What it means |
| :--- | :--- | :--- |
| **Mean Fidelity** | **0.9865** | The model is very accurate (closer to 1.0 is better). |
| **Mean Trace Distance** | **0.1025** | The error is low (closer to 0.0 is better). |
| **Inference Latency** | **1.25 ms** | It takes about 1.25 milliseconds to reconstruct one state. |

## 3. Training Analysis
The model trained quite smoothly. Using the Adam optimizer with a learning rate of 0.001, it reached a high fidelity (> 98%) within just 20 epochs. The loss curve showed steady improvement without much instability.

## 4. Conclusion
This project demonstrated that Deep Learning, specifically Transformers, can effectively perform Quantum State Tomography. The most challenging part was ensuring the physical constraints were met, but the Cholesky decomposition method worked perfectly for this. 
