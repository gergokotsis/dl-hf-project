#!/bin/bash
set -e

echo "Running data processing..."
python src/01_data_processing.py

echo "Running model training..."
python src/02_train.py

echo "Running evaluation..."
python src/03_evaluation.py

echo "Running inference..."
python src/04_inference.py
echo "Pipeline finished successfully."