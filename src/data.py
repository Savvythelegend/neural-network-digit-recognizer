"""
Data loading and preprocessing utilities.
Supports both local CSV and scikit-learn's MNIST dataset.
"""

import sys
from pathlib import Path
import numpy as np
import config


def load_mnist_data(csv_path=None, test_size=None, train_size=None):
    """
    Load MNIST data from local CSV or fetch from OpenML.

    Args:
        csv_path (Path): Path to local train.csv file.
        test_size (int): Number of validation samples.
        train_size (int): Number of training samples.

    Returns:
        tuple: (X_train, Y_train, X_val, Y_val) - normalized to [0, 1].
    """
    if train_size is None:
        train_size = config.TRAIN_SIZE
    if test_size is None:
        test_size = config.VAL_SIZE

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    if csv_path is None:
        csv_path = Path(__file__).parent / config.DATA_FILENAME
        if not csv_path.exists():
            csv_path = Path(__file__).parent / ".." / "datasets" / config.DATA_FILENAME

    try:
        # Try local CSV first
        if csv_path.exists():
            print(f"Found local dataset at: {csv_path}")
            try:
                import pandas as pd
            except ImportError:
                raise RuntimeError("pandas is required to read local train.csv. "
                                   "Install with: pip install pandas")

            df = pd.read_csv(csv_path)
            if 'label' not in df.columns:
                raise ValueError("train.csv must contain a 'label' column")

            X = df.drop('label', axis=1).values.astype(np.float32) / 255.0
            y = df['label'].values.astype(int)
        else:
            # Fallback to OpenML
            print("Local train.csv not found, fetching MNIST from OpenML...")
            try:
                from sklearn.datasets import fetch_openml
            except ImportError:
                raise RuntimeError("scikit-learn is required to fetch MNIST. "
                                   "Install with: pip install scikit-learn")

            mnist = fetch_openml('mnist_784', version=1, parser='auto')
            X = np.array(mnist.data, dtype=np.float32) / 255.0
            y = np.array(mnist.target, dtype=int)

        print(f"Total samples available: {X.shape[0]}")
        print(f"Using {train_size} training + {test_size} validation samples")

        # Train-test split
        try:
            from sklearn.model_selection import train_test_split
            X_train_full, X_val_full, Y_train, Y_val = train_test_split(
                X, y,
                train_size=min(train_size, X.shape[0] - test_size),
                test_size=min(test_size, X.shape[0] - train_size),
                random_state=config.RANDOM_SEED,
                stratify=y
            )
        except ImportError:
            # Fallback numpy-based split
            rng = np.random.RandomState(config.RANDOM_SEED)
            perm = rng.permutation(X.shape[0])
            n_train = min(train_size, X.shape[0] - test_size)
            n_val = min(test_size, max(0, X.shape[0] - n_train))
            train_idx = perm[:n_train]
            val_idx = perm[n_train:n_train + n_val]
            X_train_full = X[train_idx]
            X_val_full = X[val_idx]
            Y_train = y[train_idx]
            Y_val = y[val_idx]

        # Transpose to (features, samples) format expected by network
        X_train = X_train_full.T
        X_val = X_val_full.T

        print(f"Training set: {X_train.shape} (features × samples)")
        print(f"Validation set: {X_val.shape} (features × samples)")

        return X_train, Y_train, X_val, Y_val

    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nPlease ensure you have installed required packages:")
        print("  pip install numpy pandas scikit-learn opencv-python")
        sys.exit(1)


def preprocess_image(img):
    """
    Preprocess a 28×28 grayscale image for model input.

    Args:
        img (np.ndarray): Image array (28×28).

    Returns:
        np.ndarray: Flattened, normalized image (784, 1).
    """
    img_normalized = img.astype(np.float32) / 255.0
    img_flat = img_normalized.reshape(784, 1)
    return img_flat
