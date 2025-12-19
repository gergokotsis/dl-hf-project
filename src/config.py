import os

# =====================
# Data Preparation
# =====================
RAW_ZIP_PATH = "/data/legaltextdecoder.zip"
RAW_EXTRACT_PATH = "/data/extracted"
PROCESSED_DATA_PATH = "/data/processed_data.csv"

# =====================
# Paths
# =====================


BASE_DIR = "/app"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "log")

# =====================
# Reproducibility
# =====================
SEED = 42

# =====================
# Data
# =====================
TEST_SIZE = 0.25
VAL_SIZE = 0.40

# =====================
# TF-IDF Baseline
# =====================
TFIDF_MAX_FEATURES = 2000
TFIDF_NGRAM_RANGE = (1, 2)

# =====================
# Final Model
# =====================
MAX_VOCAB = 10000
MAX_LEN = 100
EMBEDDING_DIM = 32

# =====================
# Training
# =====================
EPOCHS_BASELINE = 20
EPOCHS_FINAL = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 5

# =====================
# Model Paths
# =====================
BASELINE_MODEL_PATH = "/app/output/baseline_model.keras"
FINAL_MODEL_PATH = "/app/output/final_model.keras"
TOKENIZER_PATH = "/app/output/tokenizer.joblib"
TFIDF_PATH = "/app/output/tfidf.joblib"
LABEL_ENCODER_PATH = "/app/output/label_encoder.joblib"