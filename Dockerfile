# Use an official Python 3.7 base image
FROM python:3.7-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the application files to the container
COPY . /app

# Install system dependencies (including BLAS/LAPACK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Pre-install to avoid AirSim/PyQt5 crashes
RUN pip install --no-cache-dir numpy==1.21.6 msgpack-rpc-python==0.4.1

# Install torch and other dependencies
RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install PyQt5 and other Python packages
RUN pip install --prefer-binary PyQt5==5.15.4 PyQt5-sip==12.13.0

# Finally install your requirements
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app/run.py"]
