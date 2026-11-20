# Base image "transformers": everything in "pytorch", plus Hugging Face transformers.
# Also used by the server to score submissions (evaluate.py needs transformers for the ATLAS reference model).
ARG BASE=ghcr.io/tue-aria/nncv-pytorch:2027
FROM ${BASE}

RUN pip install \
        transformers \
        accelerate \
        safetensors
