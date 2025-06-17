# MNIST Digit Recognizer (NumPy From Scratch)

This project implements a neural network to classify handwritten digits using the MNIST dataset. It is built entirely from scratch using **NumPy** only—without any machine learning libraries.

---

## Project Highlights

- Neural Network Architecture: **784 (input) → 64 (hidden) → 10 (output)**
- Implemented from scratch:
  - Forward Propagation
  - ReLU & Softmax Activation
  - Cross-Entropy Loss
  - Backpropagation and Gradient Descent
- Achieves **~95% accuracy** on MNIST test set

---

## Dataset

- **MNIST Handwritten Digits** (28x28 grayscale images)
- 10 output classes (digits 0 to 9)
- Source: MNIST in CSV from Kaggle(https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

---

## Project Structure

```
mnist-numpy-neuralnet/
├── Handwritten Digit Recognition (MNIST).ipynb  ← Main notebook with full pipeline
├── mnist_model.pkl                              ← Pretrained model weights
├── requirements.txt                             ← Dependencies
├── .gitignore                                   ← Files to ignore in Git
└── README.md
```
---

## Pretrained Weights

This project includes a pretrained model saved as `mnist_model.pkl`.  
You can use this to skip training and go straight to predictions.

**To load the weights:**

```python
import pickle

with open("mnist_trained_weights.pkl", "rb") as f:
    w1, b1, w2, b2 = pickle.load(f)
---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```
Then open `Handwritten Digit Recognition (MNIST).ipynb` in Jupyter Notebook.

---

## Results

- Training loss decreases steadily over epochs
- Final test accuracy: **~95.13%** using just NumPy

---

## What I Learned

- The intuition and implementation of feedforward neural networks
- How gradient descent and backpropagation work in practice
- Shape debugging and step-by-step model building

---

## Dependencies

Minimal dependencies:
```txt
numpy>=1.24
matplotlib>=3.7  # for visualizing samples
notebook>=7.0    # for running in Jupyter
```

---

Feel free to fork or contribute ideas for improvements, like:
- Mini-batch training
- Dropout regularization
- Deeper architectures

