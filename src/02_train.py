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

# ==========================================
# 1. REPRODUCIBILITY SETUP
# ==========================================
SEED = 42

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
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load processed data
df = pd.read_csv(os.path.join(OUTPUT_DIR, "processed_data.csv"))

# FORCE SORTING: Ensure consistent order before splitting
if "text" in df.columns:
    df = df.sort_values(by="text").reset_index(drop=True)

texts = df["text"].values
labels = df["label"].values

print(f"Data loaded: {len(texts)} samples")

# ==========================================
# 3. SPLITTING
# ==========================================
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.25, stratify=labels, random_state=40
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.40, stratify=y_temp, random_state=40
)

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
print("\n--- Training Baseline Model ---")
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train).toarray()
X_val_vec   = tfidf.transform(X_val).toarray()

joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "tfidf.joblib"))

baseline_model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train_vec.shape[1],)),
    Dense(32, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

baseline_model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

baseline_model.fit(
    X_train_vec, y_train_cat,
    validation_data=(X_val_vec, y_val_cat),
    epochs=20, batch_size=16, verbose=1
)
baseline_model.save(os.path.join(OUTPUT_DIR, "baseline_model.keras"))

# ==========================================
# 6. FINAL MODEL (Embedding)
# ==========================================
print("\n--- Training Final Model ---")
MAX_VOCAB = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)
joblib.dump(tokenizer, os.path.join(OUTPUT_DIR, "tokenizer.joblib"))

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
    Embedding(MAX_VOCAB, 32, input_length=MAX_LEN),
    Flatten(),
    Dense(32, activation="relu"),
    #Dropout(0.2, seed=SEED), 
    Dense(32, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

final_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

final_model.fit(
    X_train_pad, y_train_cat,
    validation_data=(X_val_pad, y_val_cat),
    epochs=25,
    batch_size=16,
    #class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

final_model.save(os.path.join(OUTPUT_DIR, "final_model.keras"))