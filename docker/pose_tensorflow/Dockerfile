FROM tensorflow/tensorflow:1.4.1-py3

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    PyYAML==3.12 \
    Cython==0.27.3 \
    scikit-image==0.13.1 \
    easydict==1.7 \
    munkres==1.0.12

WORKDIR /workspace
