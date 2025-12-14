# 1. Base image
FROM tensorflow/tensorflow:2.16.1

# 2. Working directory
WORKDIR /app

# 3. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy source code
COPY src/ .

# 6. Make run.sh executable
RUN chmod +x run.sh

# 7. Default command
CMD ["bash", "run.sh"]

