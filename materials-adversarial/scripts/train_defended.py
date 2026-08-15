import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from materials_adv.training.train import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting training for DEFENDED model...")
    model = train(
        augmented_train_path="data/processed/train_aug.csv",
        out_dir="results/models/transformer_defended"
    )
    logger.info("Defended model training complete.")
