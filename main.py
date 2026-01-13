import argparse
import logging
from src.utils import setup_logging
from src.train import train
from src.predict import DeepfakeDetector

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Momenta Audio Deepfake Detection")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    parser_train = subparsers.add_parser("train", help="Train the model")
    
    # Predict command
    parser_predict = subparsers.add_parser("predict", help="Predict on an audio file")
    parser_predict.add_argument("file", type=str, help="Path to the audio file (MP3 or FLAC)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train()
    elif args.command == "predict":
        detector = DeepfakeDetector()
        try:
            label, score = detector.predict(args.file)
            print(f"Prediction: {label} (Confidence: {score:.4f})")
        except Exception as e:
            print(f"Error: {e}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
