from pathlib import Path
import torch

# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset path
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Digital Knee X-ray Images"
    / "MedicalExpert-I"
)

# Processed data path
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"

# Class information
CLASS_NAMES = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
NUM_CLASSES = 5

LABEL_MAP = {
    0: "Normal",
    1: "Doubtful",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
}

# Image and training settings
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"