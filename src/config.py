"""
Configuration constants for the Neural Network digit recognizer.
"""

# Model Architecture
LAYER_DIMS = [10, 784, 10]  # Hidden layer size: 10, Input: 784 (28x28), Output: 10 (digits 0-9)
LEARNING_RATE = 0.1
TRAINING_ITERATIONS = 500

# Data Loading
TRAIN_SIZE = 15000
VAL_SIZE = 3000
RANDOM_SEED = 42

# Webcam & Inference
WEBCAM_ROI_RADIUS = 75
WEBCAM_CENTER_X = 320
WEBCAM_CENTER_Y = 240
THRESHOLD_INIT = 150
FRAME_SKIP_COUNT = 5

# Prediction Smoothing (for stability)
PREDICTION_SMOOTHING_WINDOW = 8  # Average predictions over 8 frames
PREDICTION_CONFIDENCE_THRESHOLD = 0.6  # Need 60% agreement to show prediction

# Image Preprocessing
MORPH_KERNEL_SIZE = 3  # Kernel size for morphological operations
USE_MORPHOLOGY = True  # Apply erosion/dilation to clean up noise

# Paths
MODEL_FILENAME = "mnist_model.npz"
DATA_FILENAME = "train.csv"

# Display
VERBOSE_TRAINING = True
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_PADDING = 80

