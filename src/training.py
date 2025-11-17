import sys
from pathlib import Path
import numpy as np
from model import NeuralNet
from data import load_mnist_data
import config


def save_model(filepath, model):
    filepath = Path(filepath)
    np.savez(
        str(filepath),
        W1=model.W1, W2=model.W2, B1=model.B1, B2=model.B2,
        layer_dims=np.array(model.layer_dims),
        learning_rate=np.array(model.learning_rate)
    )
    print(f"✓ Model saved to: {filepath}")


def load_model(filepath):
    filepath = Path(filepath)
    data = np.load(str(filepath), allow_pickle=True)
    
    layer_dims = tuple(np.array(data['layer_dims']).tolist())
    learning_rate = float(data.get('learning_rate', config.LEARNING_RATE))
    
    model = NeuralNet.from_layer_dims(list(layer_dims), learning_rate)
    model.W1 = data['W1']
    model.W2 = data['W2']
    model.B1 = data['B1']
    model.B2 = data['B2']
    
    print(f"✓ Model loaded from: {filepath}")
    return model


def get_model_path():
    return Path(__file__).parent / ".." / "model" / config.MODEL_FILENAME


def train_or_load_model(force_retrain=False):
    model_path = get_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_path.exists() and not force_retrain:
        try:
            print("\nLoading existing model...")
            return load_model(model_path)
        except Exception as e:
            print(f"Failed to load: {e}")
    
    print("\nLoading dataset...")
    X_train, Y_train, X_val, Y_val = load_mnist_data()
    
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    model = NeuralNet.from_layer_dims(config.LAYER_DIMS, config.LEARNING_RATE)
    model.fit(X_train, Y_train, num_iters=config.TRAINING_ITERATIONS, 
              verbose=config.VERBOSE_TRAINING)
    
    predictions = model.predict(X_val)
    val_accuracy = model.get_accuracy(predictions, Y_val)
    
    print("\n" + "="*60)
    print(f"✓ Training complete! Accuracy: {val_accuracy:.4f}")
    print("="*60)
    
    try:
        save_model(model_path, model)
    except Exception as e:
        print(f"Warning: Failed to save model: {e}")
    
    return model

