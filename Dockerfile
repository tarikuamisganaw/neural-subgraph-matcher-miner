FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.8 (more stable on Ubuntu 22.04)
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3.8-venv \
    python3-pip \
    build-essential \
    pkg-config \
    git \
    libfreetype6-dev \
    libpng-dev \
    libqhull-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for Python 3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.txt .

# Upgrade pip and install dependencies
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install packages with compatible versions for Python 3.8
RUN pip install --no-cache-dir \
    numpy==1.21.6 \
    matplotlib==3.5.3 \
    scikit-learn==1.0.2 \
    seaborn==0.11.2 \
    torch==1.13.1+cpu \
    torchvision==0.14.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric
RUN pip install --no-cache-dir \
    torch-scatter==2.1.1+pt113cpu \
    torch-sparse==0.6.17+pt113cpu \
    torch-geometric==2.3.1 \
    -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

# Install remaining packages
RUN pip install --no-cache-dir \
    deepsnap==0.2.2 \
    networkx==2.8.8 \
    test-tube==0.7.5 \
    tqdm==4.64.1

COPY . .