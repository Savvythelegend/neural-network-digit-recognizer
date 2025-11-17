"""
Neural Network implementation from scratch.
Includes forward propagation, backward propagation, and parameter updates.
"""

import numpy as np


class NeuralNet:
    """Two-layer neural network with ReLU hidden layer and softmax output."""

    def __init__(self, layer_dims, learning_rate=0.1):
        """
        Initialize the neural network.

        Args:
            layer_dims (list): [hidden_size, input_size, output_size]
            learning_rate (float): Learning rate for gradient descent.
        """
        self.layer_dims = tuple(layer_dims)
        self.learning_rate = learning_rate
        self.W1 = None
        self.W2 = None
        self.B1 = None
        self.B2 = None

    def init_params(self):
        """Initialize weights and biases with random values."""
        self.W1 = np.random.rand(self.layer_dims[0], self.layer_dims[1]) - 0.5
        self.B1 = np.random.rand(self.layer_dims[0], 1) - 0.5
        self.W2 = np.random.rand(self.layer_dims[2], self.layer_dims[0]) - 0.5
        self.B2 = np.random.rand(self.layer_dims[2], 1) - 0.5
        return self.W1, self.W2, self.B1, self.B2

    @staticmethod
    def relu(Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    @staticmethod
    def relu_deriv(Z):
        """Derivative of ReLU."""
        return (Z > 0).astype(float)

    @staticmethod
    def softmax(Z):
        """Softmax activation function for output layer."""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    @staticmethod
    def one_hot_encode(Y):
        """Convert class labels to one-hot encoding."""
        one_hot_Y = np.zeros((np.max(Y) + 1, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y

    def forward_prop(self, X):
        """
        Forward propagation through the network.

        Args:
            X (np.ndarray): Input data (features x samples).

        Returns:
            tuple: (Z1, A1, Z2, A2) - activations and pre-activations.
        """
        if any(v is None for v in (self.W1, self.W2, self.B1, self.B2)):
            self.init_params()

        Z1 = self.W1.dot(X) + self.B1
        A1 = self.relu(Z1)
        Z2 = self.W2.dot(A1) + self.B2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def backward_prop(self, Z1, A1, Z2, A2, X, Y):
        """
        Backward propagation to compute gradients.

        Args:
            Z1, A1, Z2, A2: Forward propagation outputs.
            X (np.ndarray): Input data.
            Y (np.ndarray): Labels (class indices).

        Returns:
            tuple: (dW1, dB1, dW2, dB2) - gradients for each parameter.
        """
        m = X.shape[1]
        one_hot_Y = self.one_hot_encode(Y)

        dZ2 = A2 - one_hot_Y
        dW2 = (1.0 / m) * dZ2.dot(A1.T)
        dB2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = self.W2.T.dot(dZ2) * self.relu_deriv(Z1)
        dW1 = (1.0 / m) * dZ1.dot(X.T)
        dB1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, dB1, dW2, dB2

    def update_params(self, dW1, dB1, dW2, dB2):
        """Update parameters using gradient descent."""
        self.W1 -= self.learning_rate * dW1
        self.B1 -= self.learning_rate * dB1
        self.W2 -= self.learning_rate * dW2
        self.B2 -= self.learning_rate * dB2

    def fit(self, X, Y, num_iters=1000, verbose=True):
        """
        Train the network on data.

        Args:
            X (np.ndarray): Training data (features x samples).
            Y (np.ndarray): Training labels.
            num_iters (int): Number of iterations.
            verbose (bool): Print progress.
        """
        for i in range(num_iters):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            dW1, dB1, dW2, dB2 = self.backward_prop(Z1, A1, Z2, A2, X, Y)
            self.update_params(dW1, dB1, dW2, dB2)

            if verbose and (i % 50) == 0:
                predictions = self.get_predictions(A2)
                accuracy = self.get_accuracy(predictions, Y)
                print(f"  Iteration {i:5d} - Accuracy: {accuracy:.4f}")

    def predict(self, X):
        """
        Make predictions on input data.

        Args:
            X (np.ndarray): Input data (features x samples).

        Returns:
            np.ndarray: Predicted class indices.
        """
        _, _, _, A2 = self.forward_prop(X)
        return self.get_predictions(A2)

    @staticmethod
    def get_predictions(A2):
        """Get class predictions from softmax output."""
        return np.argmax(A2, axis=0)

    @staticmethod
    def get_accuracy(predictions, Y):
        """Compute accuracy of predictions."""
        return np.sum(predictions == Y) / Y.size

    @classmethod
    def from_layer_dims(cls, layer_dims, learning_rate=0.1):
        """Factory method to create and initialize a network."""
        net = cls(layer_dims, learning_rate=learning_rate)
        net.init_params()
        return net
