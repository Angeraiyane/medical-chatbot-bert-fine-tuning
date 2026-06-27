from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossiers
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Modèle BERT
MODEL_NAME = "bert-base-uncased"

# Hyperparamètres
MAX_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
RANDOM_STATE = 42