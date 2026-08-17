import argparse
from materials_adv.training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/models/transformer_regressor")
    parser.add_argument("--scaler-path", type=str, default=None)
    args = parser.parse_args()
    
    train(augmented_train_path=args.augmented, out_dir=args.out_dir, scaler_path=args.scaler_path)
