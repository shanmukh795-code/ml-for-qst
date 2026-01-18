# Part 2: Replication Guide

This is a step-by-step guide on how to set up my project and recreate the results I got.

## 1. Setup Instructions

First, you'll need to make sure you have Python installed. I used Python 3.8+ for this project.

You can install all the necessary libraries (like PyTorch and NumPy) using the `requirements.txt` file I included. Just open your terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

## 2. Generating Data
I wrote a script `src/data.py` that generates random quantum states and simulates measurements on them. You don't usually need to run this separately because the training script does it automatically on the fly, but if you want to test it, you can run:

```bash
python src/data.py
```

## 3. Training the Model
To actually train the neural network, I created a `train.py` script. This will:
1. Generate a dataset of random density matrices.
2. Train the Transformer model for 20 epochs.
3. Save the best model to the `outputs/` folder.

Run it with:
```bash
python src/train.py
```

You should see the Loss going down and the Validation Fidelity going up in the terminal output.

## 4. Evaluation
To get the final numbers for the report, I made a separate evaluation script. This loads the saved model and tests it on a fresh batch of 1000 random states to check how well it generalizes.

Run:
```bash
python src/evaluate.py
```

This will print out the **Mean Fidelity**, **Trace Distance**, and the **Inference Time** (latency) per sample.

## 5. Folder Structure
Here is how I organized my code:
- **`src/`**: Contains all the python code.
    - `data.py`: Handles the quantum simulation.
    - `model.py`: Where the Transformer architecture is defined.
    - `train.py`: The main loop for learning.
    - `utils.py`: Helper functions for math (like calc_fidelity).
- **`outputs/`**: Where the trained model gets saved.
- **`docs/`**: Documentation files (this guide and the model explanation).
