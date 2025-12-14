import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Setup
# -----------------------------
OUTPUT_PATH = "/app/output"
os.makedirs(OUTPUT_PATH, exist_ok=True)
report_path = os.path.join(OUTPUT_PATH, "evaluation_report.txt")
summary_path = os.path.join(OUTPUT_PATH, "evaluation_summary.csv")

# Initialize file
report_file = open(report_path, "w", encoding="utf-8")

def log_to_file(text):
    print(text)  # Print to console
    report_file.write(text + "\n")

# -----------------------------
# Load Data & Encoders
# -----------------------------
print("Loading data and encoders...")
label_encoder = joblib.load(os.path.join(OUTPUT_PATH, "label_encoder.joblib"))
X_test, y_test = joblib.load(os.path.join(OUTPUT_PATH, "test_split.joblib"))

# Prepare labels for Keras (needed for Loss calculation)
y_test_int = label_encoder.transform(y_test)
NUM_CLASSES = len(label_encoder.classes_)
y_test_cat = to_categorical(y_test_int, NUM_CLASSES)

# -----------------------------
# Helper Function: Evaluate Keras Model
# -----------------------------
def evaluate_keras_model(model, X_input, y_true_cat, y_true_int, classes, name):
    log_to_file(f"\n{'='*40}")
    log_to_file(f"EVALUATION: {name}")
    log_to_file(f"{'='*40}")

    # 1. Get exact Loss and Accuracy from Keras (matches training output)
    loss, acc = model.evaluate(X_input, y_true_cat, verbose=0)
    log_to_file(f"Test Accuracy: {acc:.4f}")
    log_to_file(f"Test Loss:     {loss:.4f}\n")

    # 2. Get predictions for Classification Report
    y_pred_probs = model.predict(X_input, verbose=0)
    y_pred_int = np.argmax(y_pred_probs, axis=1)

    # 3. Generate Report
    report = classification_report(
        y_true_int, 
        y_pred_int, 
        target_names=classes,
        digits=4 # ensures high precision in the table
    )
    
    log_to_file("Classification Report:")
    log_to_file(report)
    
    return acc, loss

# -----------------------------
# 1. Random Baseline (Benchmark)
# -----------------------------
log_to_file(f"\n{'='*40}")
log_to_file(f"EVALUATION: RANDOM BASELINE")
log_to_file(f"{'='*40}")

# Generate random predictions based on training distribution
y_train = joblib.load(os.path.join(OUTPUT_PATH, "train_labels.joblib"))
label_counts = Counter(y_train)
labels_unique = list(label_counts.keys())
probs = np.array(list(label_counts.values()), dtype=float)
probs /= probs.sum()

random_preds = np.random.choice(labels_unique, size=len(y_test), p=probs)
random_acc = accuracy_score(y_test, random_preds)

log_to_file(f"Test Accuracy: {random_acc:.4f}")
log_to_file(f"Test Loss:     N/A (Random)\n")
log_to_file("Classification Report:")
log_to_file(classification_report(y_test, random_preds, digits=4))


# -----------------------------
# 2. TF-IDF Baseline
# -----------------------------
baseline_model = load_model(os.path.join(OUTPUT_PATH, "baseline_model.keras"))
tfidf = joblib.load(os.path.join(OUTPUT_PATH, "tfidf.joblib"))

X_test_vec = tfidf.transform(X_test).toarray()

base_acc, base_loss = evaluate_keras_model(
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
final_model = load_model(os.path.join(OUTPUT_PATH, "final_model.keras"))
tokenizer = joblib.load(os.path.join(OUTPUT_PATH, "tokenizer.joblib"))

MAX_LEN = 100
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN) # Use default padding (pre) to match training usually, or specific if defined

final_acc, final_loss = evaluate_keras_model(
    final_model, 
    X_test_pad, 
    y_test_cat, 
    y_test_int, 
    label_encoder.classes_, 
    "FINAL EMBEDDING MODEL"
)

# -----------------------------
# Save Summary CSV
# -----------------------------
summary_df = pd.DataFrame({
    "model": ["Random", "TF-IDF Baseline", "Final Model"],
    "accuracy": [random_acc, base_acc, final_acc],
    "loss": [None, base_loss, final_loss]
})
summary_df.to_csv(summary_path, index=False)

report_file.close()
print(f"\nFull evaluation report saved to: {report_path}")
print(f"Summary metrics saved to: {summary_path}")