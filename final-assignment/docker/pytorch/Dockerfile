# Base image "pytorch": the environment only, no course code.
# Used for training and evaluation on Snellius (with Apptainer) and for submissions on the challenge server,
# where the server mounts the fixed harness and your zip, and the job runs without internet.
FROM pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime

# links the package on the GitHub Container Registry to this repository (inherited by the other images)
LABEL org.opencontainers.image.source="https://github.com/TUE-ARIA/NNCV" \
      org.opencontainers.image.description="5LSM0 base image: PyTorch, torchvision, timm, W&B"

# PyTorch lives in the system Python of this image, which blocks pip by default (PEP 668).
# Inside a container that protection has no purpose, so pip may install into it, also for students' wheels.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1

# torchvision matching the installed PyTorch and CUDA, if the image doesn't include it already
RUN python -c "import torchvision" 2>/dev/null || \
    pip install "torch==$(python -c 'import torch; print(torch.__version__.split("+")[0])')" torchvision \
        --index-url https://download.pytorch.org/whl/cu132

RUN pip install \
        timm \
        numpy \
        pillow \
        pyyaml \
        scipy \
        opencv-python-headless \
        wandb

# fail the build here, not in a student's job, if anything is missing
RUN python -c "import torch, torchvision, timm, cv2, scipy, yaml, wandb; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CUDA', torch.version.cuda)"
WORKDIR /workspace
