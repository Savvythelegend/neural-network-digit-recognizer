"""
Main entry point for the digit recognizer application.
Coordinates training and inference pipelines.
"""

import sys
from pathlib import Path
from training import train_or_load_model
from inference import WebcamRecognizer


def main():
    """Run the full digit recognizer pipeline."""
    print("\n" + "=" * 60)
    print("NEURAL NETWORK DIGIT RECOGNIZER")
    print("=" * 60)

    # Step 1: Train or load model
    print("\n[Step 1/2] Preparing model...")
    model = train_or_load_model()

    # Step 2: Start inference
    print("\n[Step 2/2] Starting webcam interface...")
    recognizer = WebcamRecognizer(model)
    
    try:
        recognizer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

    print("\n✓ Application closed successfully.")


if __name__ == "__main__":
    main()
