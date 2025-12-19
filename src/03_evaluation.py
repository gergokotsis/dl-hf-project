import os
import joblib
import numpy as np
from collections import Counter

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from src import config
from src.utils import get_logger

# -----------------------------
# Setup
# -----------------------------
logger = get_logger("evaluation")
logger.info("===== STARTING MODEL EVALUATION =====")

OUTPUT_DIR = config.OUTPUT_DIR

# -----------------------------
# Load Data & Encoders
# -----------------------------
logger.info("Loading test data and encoders...")

label_encoder = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
X_test, y_test = joblib.load(os.path.join(OUTPUT_DIR, "test_split.joblib"))

y_test_int = label_encoder.transform(y_test)
NUM_CLASSES = len(label_encoder.classes_)
y_test_cat = to_categorical(y_test_int, NUM_CLASSES)

logger.info(f"Test samples: {len(y_test)}")
logger.info(f"Classes: {list(label_encoder.classes_)}")

# -----------------------------
# Helper Function
# -----------------------------
def evaluate_keras_model(model, X_input, y_true_cat, y_true_int, classes, name):
    logger.info("=" * 50)
    logger.info(f"EVALUATION: {name}")
    logger.info("=" * 50)

    loss, acc = model.evaluate(X_input, y_true_cat, verbose=0)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test Loss:     {loss:.4f}")

    y_pred_probs = model.predict(X_input, verbose=0)
    y_pred_int = np.argmax(y_pred_probs, axis=1)

    logger.info("Classification Report:")
    logger.info("\n" + classification_report(
        y_true_int,
        y_pred_int,
        target_names=classes,
        digits=4
    ))

    cm = confusion_matrix(y_true_int, y_pred_int)
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")

    return acc, loss

# -----------------------------
# 1. Random Baseline
# -----------------------------
logger.info("=" * 50)
logger.info("EVALUATION: RANDOM BASELINE")
logger.info("=" * 50)

y_train = joblib.load(os.path.join(OUTPUT_DIR, "train_labels.joblib"))
label_counts = Counter(y_train)

labels_unique = list(label_counts.keys())
probs = np.array(list(label_counts.values()), dtype=float)
probs /= probs.sum()

random_preds = np.random.choice(labels_unique, size=len(y_test), p=probs)
random_acc = accuracy_score(y_test, random_preds)

logger.info(f"Test Accuracy: {random_acc:.4f}")
logger.info("Classification Report:")
logger.info("\n" + classification_report(y_test, random_preds, digits=4))
logger.info("Confusion Matrix:")
logger.info(f"\n{confusion_matrix(y_test, random_preds)}")

# -----------------------------
# 2. TF-IDF Baseline
# -----------------------------
baseline_model = load_model(config.BASELINE_MODEL_PATH)
tfidf = joblib.load(os.path.join(OUTPUT_DIR, "tfidf.joblib"))

baseline_model.trainable = False

X_test_vec = tfidf.transform(X_test).toarray()

evaluate_keras_model(
    baseline_model,
    X_test_vec,
    y_test_cat,
    y_test_int,
    label_encoder.classes_,
    "TF-IDF BASELINE"
)

# -----------------------------
# 3. Final Embedding Model
# -----------------------------
final_model = load_model(config.FINAL_MODEL_PATH)
tokenizer = joblib.load(config.TOKENIZER_PATH)

X_test_pad = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=config.MAX_LEN
)

final_model.trainable = False

evaluate_keras_model(
    final_model,
    X_test_pad,
    y_test_cat,
    y_test_int,
    label_encoder.classes_,
    "FINAL EMBEDDING MODEL"
)

logger.info("===== EVALUATION COMPLETED SUCCESSFULLY =====")
