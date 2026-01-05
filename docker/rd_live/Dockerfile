# Dockerfile for rd_live.py
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy feature_store package
COPY feature_store/ /app/feature_store/

# Create directory for logs and data
RUN mkdir -p /app/logs /app/data

# Run the script directly
CMD ["python", "-m", "feature_store.rd_live"]

