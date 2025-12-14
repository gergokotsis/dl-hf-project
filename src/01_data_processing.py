import zipfile
import os
import json
import re
import pandas as pd

ZIP_PATH = "/data/legaltextdecoder.zip"
EXTRACT_PATH = "/data/extracted"
OUTPUT_PATH = "/app/output/processed_data.csv"

os.makedirs("/app/output", exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

all_data = []

for root, dirs, files in os.walk(EXTRACT_PATH):
    dirs[:] = [d for d in dirs if d != "consensus"]
    files = sorted(files)
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)

texts = []
labels = []

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

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

clean_texts = [clean_text(t) for t in texts]

df = pd.DataFrame({
    "text": clean_texts,
    "label": labels
})

print(df['label'].value_counts())

df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

print(f"Saved {len(df)} samples to {OUTPUT_PATH}")
