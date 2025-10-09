FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.7 from Ubuntu's default repositories (more stable)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    && apt-get update \
    && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    python3.7-venv \
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

# Install pip for Python 3.7
RUN curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py \
    && python3.7 get-pip.py \
    && rm get-pip.py

# Create symlinks
RUN ln -sf /usr/bin/python3.7 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.7 /usr/bin/pip

WORKDIR /app

COPY requirements.txt .

# Upgrade pip and setuptools first
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first (foundation for other packages)
RUN pip install --no-cache-dir numpy==1.21.6

# Install other dependencies
RUN pip install --no-cache-dir \
    matplotlib==3.5.3 \
    scikit-learn==1.0.2 \
    seaborn==0.11.2

# Install PyTorch with specific versions that work with Python 3.7
RUN pip install --no-cache-dir \
    torch==1.13.1+cpu \
    torchvision==0.14.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric dependencies
RUN pip install --no-cache-dir \
    torch-scatter==2.1.1+pt113cpu \
    torch-sparse==0.6.17+pt113cpu \
    torch-cluster==1.6.1+pt113cpu \
    torch-spline-conv==1.2.2+pt113cpu \
    torch-geometric==2.3.1 \
    -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

# Install remaining packages
RUN pip install --no-cache-dir \
    deepsnap==0.2.2 \
    networkx==2.8.8 \
    test-tube==0.7.5 \
    tqdm==4.64.1

COPY . .