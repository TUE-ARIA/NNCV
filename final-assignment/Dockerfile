# Submission image. Build from the `final-assignment` folder, choosing your track:
#   docker build --build-arg TRACK=consistency -t nncv-consistency .
# TRACK is one of: segmentation, consistency, realtime, generative.
# Your weights must be at `final-assignment/model.pt`.
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ARG TRACK
RUN test -n "$TRACK" || (echo "Pass --build-arg TRACK=<track>" && exit 1)
ENV TRACK=${TRACK} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Add any extra packages your predictor needs (the server has no internet access at run time).
RUN pip install --no-cache-dir pillow numpy

WORKDIR /app
# Copies the code and model.pt; data, checkpoints and logs are excluded by .dockerignore.
COPY . /app

ENTRYPOINT ["python", "-m", "submission.run"]
