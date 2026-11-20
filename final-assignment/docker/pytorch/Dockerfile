# Base image "pytorch" for challenge submissions: the environment only, no course code.
# The server mounts the fixed harness and your zip; the job runs without internet.
FROM pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install \
        timm \
        numpy \
        pillow \
        pyyaml \
        scipy \
        opencv-python-headless
WORKDIR /workspace
