
# Use an official Python runtime as a parent image
# Utilizing a slim version for smaller size, but full version ensures all C bindings (like for torchaudio) are present
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Memory Optimization for Render Free Tier (512MB Limit)
# These prevent PyTorch/NumPy from spawning multiple threads which eats RAM individually
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV PYTORCH_JIT=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Optimize for Cloud size: Install CPU-only PyTorch first (Huge space saver)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the rest
COPY requirements.txt /app/
# Remove torch from requirements.txt to avoid reinstalling huge version
# (We handle this by assuming requirements.txt contains 'torch' but pip encounters it as satisfied or we use grep to filter)
# A safer way is to rely on pip satisfying the requirement with the installed cpu version.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Make port 8501 available to the world outside this container (Streamlit default)
EXPOSE 8501

# Define environment variable
ENV MODEL_SAVE_PATH="models/cnn_gru_model.pth"

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
