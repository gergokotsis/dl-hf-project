import os
import joblib
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src import config
from src.utils import get_logger

# -----------------------------
# Setup
# -----------------------------
logger = get_logger("inference")
logger.info("===== STARTING INFERENCE =====")

OUTPUT_DIR = config.OUTPUT_DIR

# -----------------------------
# Load Shared Artifacts
# -----------------------------
logger.info("Loading label encoder...")
label_encoder = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

# -----------------------------
# Inference Functions
# -----------------------------
def infer_baseline(texts):
    """
    Run inference using the TF-IDF baseline model.
    """
    logger.info("Running inference with TF-IDF baseline model")

    tfidf = joblib.load(os.path.join(OUTPUT_DIR, "tfidf.joblib"))
    model = load_model(config.BASELINE_MODEL_PATH)
    model.trainable = False

    X_vec = tfidf.transform(texts).toarray()
    probs = model.predict(X_vec, verbose=0)
    preds = np.argmax(probs, axis=1)
    labels = label_encoder.inverse_transform(preds)

    for text, label, prob in zip(texts, labels, probs):
        logger.info("-" * 40)
        logger.info(f"TEXT: {text}")
        logger.info(f"PREDICTED LABEL: {label}")
        logger.info(f"CONFIDENCE: {np.max(prob):.4f}")

    return labels


def infer_final(texts):
    """
    Run inference using the final embedding-based model.
    """
    logger.info("Running inference with final embedding model")

    tokenizer = joblib.load(config.TOKENIZER_PATH)
    model = load_model(config.FINAL_MODEL_PATH)
    model.trainable = False

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=config.MAX_LEN)

    probs = model.predict(padded, verbose=0)
    preds = np.argmax(probs, axis=1)
    labels = label_encoder.inverse_transform(preds)

    for text, label, prob in zip(texts, labels, probs):
        logger.info("-" * 40)
        logger.info(f"TEXT: {text}")
        logger.info(f"PREDICTED LABEL: {label}")
        logger.info(f"CONFIDENCE: {np.max(prob):.4f}")

    return labels


# -----------------------------
# Example Inference Run
# -----------------------------
example_texts = [
    "A Felek közötti megállapodást a jelen ÁSZF, a Szolgáltatási Szerződés és ezek mellékletei együttesen tartalmazzák. Bármilyen, a jelen ÁSZF, valamint a Szolgáltatási Szerződés és annak mellékletei eltérése esetén elsősorban a Szolgáltatási Szerződés és másodsorban a jelen ÁSZF rendelkezései az irányadóak.",
    "Az Ügyfél köteles a Szolgáltatás igénybevételekor az erre vonatkozó jogszabályokat és a Szolgáltató működési rendjét tiszteletben tartani. Mindemellett az Ügyfél a Szolgáltatás igénybevételekor köteles tiszteletben tartani az erre vonatkozó jogszabályi rendelkezéseket. Az Ügyfél köteles a Szolgáltatások nyújtásában a Szolgáltatóval illetőleg annak közreműködőivel, az orvossal, illetve az egyéb alkalmazottakkal együttműködni, ellenkező esetben a Szolgáltató jogosult megtagadni a Szolgáltatás nyújtását, melynek következményei az Ügyfelet terhelik."
]

logger.info("===== BASELINE MODEL PREDICTIONS =====")
infer_baseline(example_texts)

logger.info("===== FINAL MODEL PREDICTIONS =====")
infer_final(example_texts)

logger.info("===== INFERENCE COMPLETED SUCCESSFULLY =====")


