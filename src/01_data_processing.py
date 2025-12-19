import zipfile
import os
import json
import re
import requests
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger("data_preparation")

# ---------------------------------------------------------------------
# 1. Download ZIP from SharePoint
# ---------------------------------------------------------------------
SHAREPOINT_URL = (
    "https://bmeedu-my.sharepoint.com/:u:/g/personal/"
    "gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I"
    "?e=iFp3iz&download=1"
)

ZIP_PATH = config.RAW_ZIP_PATH
EXTRACT_PATH = config.RAW_EXTRACT_PATH
OUTPUT_PATH = config.PROCESSED_DATA_PATH

os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
os.makedirs(EXTRACT_PATH, exist_ok=True)

logger.info("===== STARTING DATA PREPARATION PIPELINE =====")

def download_zip():
    logger.info(f"Downloading dataset from SharePoint...")

    try:
        response = requests.get(SHAREPOINT_URL, timeout=60)

        if response.status_code != 200:
            logger.error(f"Download failed. HTTP {response.status_code}")
            return False

        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)

        logger.info(f"Dataset successfully downloaded → {ZIP_PATH}")
        return True

    except Exception as e:
        logger.error(f"Exception during download: {e}")
        return False


# ---------------------------------------------------------------------
# 2. Extract ZIP
# ---------------------------------------------------------------------
def extract_zip():
    logger.info(f"Extracting ZIP file from {ZIP_PATH}")

    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)

        logger.info(f"ZIP extraction completed → {EXTRACT_PATH}")
        return True

    except zipfile.BadZipFile:
        logger.error("The downloaded file is not a valid ZIP archive.")
        return False

    except Exception as e:
        logger.error(f"Error during ZIP extraction: {e}")
        return False


# ---------------------------------------------------------------------
# 3. Download + Extract before parsing files
# ---------------------------------------------------------------------
if not os.path.exists(ZIP_PATH):
    download_ok = download_zip()
    if not download_ok:
        logger.error("Stopping: Cannot continue without dataset ZIP.")
        exit(1)

extract_ok = extract_zip()
if not extract_ok:
    logger.error("Stopping: ZIP extraction failed.")
    exit(1)

# ---------------------------------------------------------------------
# 4. Parse JSON files
# ---------------------------------------------------------------------
logger.info("Parsing JSON files...")
all_data = []

for root, dirs, files in os.walk(EXTRACT_PATH):
    dirs[:] = [d for d in dirs if d != "consensus"]
    files = sorted(files)
    for file in files:
        if file.endswith(".json"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                logger.error(f"Failed to read {filepath}: {e}")

logger.info(f"Total raw JSON samples loaded: {len(all_data)}")

# ---------------------------------------------------------------------
# 5. Extract text + labels
# ---------------------------------------------------------------------
texts, labels = [], []

for item in all_data:
    text = item.get("data", {}).get("text")
    annotations = item.get("annotations", [])

    if not text or not annotations:
        continue

    results = annotations[0].get("result", [])
    if not results:
        continue

    choices = results[0].get("value", {}).get("choices", [])
    if not choices:
        continue

    texts.append(text)
    labels.append(choices[0])


# ---------------------------------------------------------------------
# 6. Clean text
# ---------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

clean_texts = [clean_text(t) for t in texts]

df = pd.DataFrame({
    "text": clean_texts,
    "label": labels
})

logger.info(f"Samples after cleaning: {len(df)}")

# ---------------------------------------------------------------------
# 7. Data analysis logs
# ---------------------------------------------------------------------
duplicate_count = df.duplicated(subset="text").sum()
logger.info(f"Duplicate texts detected: {duplicate_count}")

df = df.drop_duplicates(subset="text")
logger.info(f"Samples after removing duplicates: {len(df)}")

logger.info("Label distribution:")
logger.info(f"\n{df['label'].value_counts()}")

df["text_length"] = df["text"].apply(lambda x: len(x.split()))

logger.info("Text length statistics:")
logger.info("\n" + df["text_length"].describe().to_string())

label_counts = df["label"].value_counts()
imbalance_ratio = label_counts.max() / label_counts.min()
logger.info("Class imbalance analysis:")
logger.info(f"\n{label_counts}")
logger.info(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")

short_texts = (df["text_length"] < 5).sum()
logger.info(f"Texts with fewer than 5 words: {short_texts}")

# ---------------------------------------------------------------------
# 8. Save processed dataset
# ---------------------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
logger.info(f"Saved {len(df)} processed samples → {OUTPUT_PATH}")

logger.info("===== DATA PREPARATION COMPLETED SUCCESSFULLY =====")
