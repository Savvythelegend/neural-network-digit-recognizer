# Neural Network Digit Recognizer

A modular, from-scratch neural network trained on MNIST with real-time webcam inference. Pure NumPy—no frameworks!

![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Quick Start

```bash
pip install -r requirements.txt
cd src
python main.py
```

## Features

- 2-layer neural network (ReLU + Softmax)
- Real-time webcam inference
- Prediction smoothing (no flickering)
- Model checkpointing
- ~87-88% accuracy on MNIST

## Architecture Breakdown

**Network Structure:**
```
Input Layer (784 neurons: 28×28 pixels)
    ↓
Hidden Layer (10 neurons, ReLU activation)
    ↓
Output Layer (10 neurons, Softmax activation)
    ↓
Prediction (digit 0-9)
```

**Layer Details:**

| Layer | Type | Neurons | Activation | Purpose |
|-------|------|---------|-----------|---------|
| Input | Input | 784 | — | Flattened 28×28 MNIST image |
| Hidden | Dense | 10 | ReLU | Feature extraction |
| Output | Dense | 10 | Softmax | Digit classification (0-9) |

**Training Parameters:**
- Dataset: MNIST (15,000 training samples)
- Optimizer: Stochastic Gradient Descent (lr=0.1)
- Epochs: 500 iterations
- Accuracy: ~88% on validation set

## Controls

| Key | Action |
|-----|--------|
| `i` | Toggle inference ON/OFF |
| `q` | Quit application |
| **Threshold slider** | Adjust binary threshold |

**Tips:** Write on white paper with dark marker; ensure good lighting

## Project Structure

```
src/
├── main.py           # Entry point
├── config.py         # Configuration constants
├── model.py          # NeuralNet class
├── data.py           # Data loading & preprocessing
├── training.py       # Training pipeline
└── inference.py      # Webcam interface
```

## Usage

### Retrain Model
```python
from training import train_or_load_model
model = train_or_load_model(force_retrain=True)
```

### Single Prediction
```python
from training import load_model
from inference import predict_digit
import cv2

model = load_model("../model/mnist_model.npz")
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
print(f"Predicted: {predict_digit(model, img)}")
```

## 📚 Learning Resources

### Neural Network Fundamentals
- **Vizuara - Neural Network from Scratch**  
  https://www.youtube.com/watch?v=A83BbHFoKb8  
  Complete guide to building neural networks from scratch, forward propagation, and network architecture.

### Backpropagation & Gradient Descent
- **Dr. RC - Computer Science (Backpropagation Playlist)**  
  https://www.youtube.com/playlist?list=PLJ4-ETiGBrdOZuDqcuEkGH6_MqGoxy4HW  
  In-depth explanation of backpropagation, gradient computation, and optimization techniques.

---

**Built from scratch** 🎓 | **Pure NumPy** 📐 | **Production Ready** ✨
