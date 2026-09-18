import os
import torch
import pandas as pd
from model import MotifCNN
from attack import generate_adversarial_dataset

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "models", "defended_model.pth")
    data_path = os.path.join(script_dir, "..", "data", "raw", "dataset.csv")
    output_path = os.path.join(script_dir, "..", "data", "raw", "adv_dataset_against_defended.csv")
    
    print("Evaluating attack against DEFENDED model...")
    generate_adversarial_dataset(model_path, data_path, output_path, max_iters=10)
