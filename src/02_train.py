import os
import random
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from src import config
from src.utils import get_logger
from tensorflow.keras.callbacks import Callback

class EpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.info(
            f"Epoch {epoch+1} | "
            f"loss={logs['loss']:.4f} | "
            f"acc={logs['accuracy']:.4f} | "
            f"val_loss={logs.get('val_loss', 0):.4f} | "
            f"val_acc={logs.get('val_accuracy', 0):.4f}"
        )

logger = get_logger("training")

logger.info("===== TRAINING CONFIGURATION =====")
for k, v in vars(config).items():
    if k.isupper():
        logger.info(f"{k}: {v}")

def log_model_summary(model, name):
    logger.info(f"===== MODEL ARCHITECTURE: {name} =====")
    model.summary(print_fn=lambda x: logger.info(x))


# ==========================================
# 1. REPRODUCIBILITY SETUP
# ==========================================

SEED = config.SEED

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seeds()

# ==========================================
# 2. PATHS & DATA LOADING
# ==========================================

OUTPUT_DIR = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load processed data
df = pd.read_csv(config.PROCESSED_DATA_PATH)

# FORCE SORTING: Ensure consistent order before splitting
if "text" in df.columns:
    df = df.sort_values(by="text").reset_index(drop=True)

texts = df["text"].values
labels = df["label"].values

logger.info(f"Data loaded: {len(texts)} samples")

# ==========================================
# 3. SPLITTING
# ==========================================
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.25, stratify=labels, random_state=40
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.40, stratify=y_temp, random_state=40
)

logger.info("Dataset split completed:")
logger.info(f"Train samples: {len(X_train)}")
logger.info(f"Validation samples: {len(X_val)}")
logger.info(f"Test samples: {len(X_test)}")

# Save splits
np.savez(
    os.path.join(OUTPUT_DIR, "splits.npz"),
    X_train=X_train, X_val=X_val, X_test=X_test,
    y_train=y_train, y_val=y_val, y_test=y_test
)
joblib.dump(y_train, os.path.join(OUTPUT_DIR, "train_labels.joblib"))
joblib.dump((X_test, y_test), os.path.join(OUTPUT_DIR, "test_split.joblib"))
joblib.dump(X_train, os.path.join(OUTPUT_DIR, "texts.joblib"))

# ==========================================
# 4. ENCODING
# ==========================================
label_encoder = LabelEncoder()
label_encoder.fit(labels)
joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

NUM_CLASSES = len(label_encoder.classes_)

y_train_int = label_encoder.transform(y_train)
y_val_int   = label_encoder.transform(y_val)
y_test_int  = label_encoder.transform(y_test) # Prepared for evaluation

y_train_cat = to_categorical(y_train_int, NUM_CLASSES)
y_val_cat   = to_categorical(y_val_int, NUM_CLASSES)
y_test_cat  = to_categorical(y_test_int, NUM_CLASSES) # Prepared for evaluation

# ==========================================
# 5. BASELINE MODEL (TF-IDF)
# ==========================================
logger.info("===== BASELINE MODEL HYPERPARAMETERS =====")
logger.info(f"TFIDF_MAX_FEATURES: {config.TFIDF_MAX_FEATURES}")
logger.info(f"TFIDF_NGRAM_RANGE: {config.TFIDF_NGRAM_RANGE}")
logger.info(f"EPOCHS: {config.EPOCHS_BASELINE}")
logger.info(f"BATCH_SIZE: {config.BATCH_SIZE}")
logger.info(f"LEARNING_RATE: {config.LEARNING_RATE}")



tfidf = TfidfVectorizer(
    max_features=config.TFIDF_MAX_FEATURES,
    ngram_range=config.TFIDF_NGRAM_RANGE
)
X_train_vec = tfidf.fit_transform(X_train).toarray()
X_val_vec   = tfidf.transform(X_val).toarray()

joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "tfidf.joblib"))

baseline_model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train_vec.shape[1],)),
    Dense(32, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

baseline_model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

log_model_summary(baseline_model, "TF-IDF BASELINE")

logger.info("Starting baseline model training...")

baseline_model.fit(
    X_train_vec, y_train_cat,
    validation_data=(X_val_vec, y_val_cat),
    epochs=config.EPOCHS_BASELINE,
    batch_size=config.BATCH_SIZE, 
    callbacks=[EpochLogger()],
    verbose=0
)

baseline_model.save(config.BASELINE_MODEL_PATH)
logger.info(f"Baseline model saved to {config.BASELINE_MODEL_PATH}")

# ==========================================
# 6. FINAL MODEL (Embedding)
# ==========================================
logger.info("===== FINAL MODEL HYPERPARAMETERS =====")
logger.info(f"MAX_VOCAB: {config.MAX_VOCAB}")
logger.info(f"MAX_LEN: {config.MAX_LEN}")
logger.info(f"EMBEDDING_DIM: {config.EMBEDDING_DIM}")
logger.info(f"EPOCHS: {config.EPOCHS_FINAL}")
logger.info(f"BATCH_SIZE: {config.BATCH_SIZE}")
logger.info(f"LEARNING_RATE: {config.LEARNING_RATE}")
logger.info(f"EARLY_STOP_PATIENCE: {config.EARLY_STOP_PATIENCE}")



MAX_VOCAB = config.MAX_VOCAB
MAX_LEN = config.MAX_LEN

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)
joblib.dump(tokenizer, config.TOKENIZER_PATH)
logger.info(f"Tokenizer saved to {config.TOKENIZER_PATH}")

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
X_val_pad   = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=MAX_LEN)
# Prepare Test Data for Final Model
X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weights = dict(enumerate(class_weights))

final_model = Sequential([
    Embedding(MAX_VOCAB, config.EMBEDDING_DIM),
    Flatten(),
    Dense(32, activation="relu"),
    #Dropout(0.2, seed=SEED), 
    Dense(32, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

final_model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

final_model.build(input_shape=(None, MAX_LEN))

log_model_summary(final_model, "FINAL EMBEDDING MODEL")

logger.info("Starting final model training...")

final_model.fit(
    X_train_pad, y_train_cat,
    validation_data=(X_val_pad, y_val_cat),
    epochs=config.EPOCHS_FINAL,
    batch_size=config.BATCH_SIZE,
    #class_weight=class_weights,
    callbacks=[EpochLogger(), early_stop],
    verbose=0
)

final_model.save(config.FINAL_MODEL_PATH)
logger.info(f"Final model saved to {config.FINAL_MODEL_PATH}")