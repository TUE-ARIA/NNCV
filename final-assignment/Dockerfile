# Complete submission image, for advanced students only: most submissions are a zip
# (python -m submission.package). Build from the `final-assignment` folder:
#   docker build --build-arg TRACK=consistency -t my-submission .
#   docker save -o image.tar my-submission
# The server runs its own harness in this image, with your code in /app and your weights in /app/model.pt.
FROM ghcr.io/tue-aria/nncv-pytorch:2027

ARG TRACK
RUN test -n "$TRACK" || (echo "Pass --build-arg TRACK=<track>" && exit 1)
ENV TRACK=${TRACK} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Add the extra packages your predictor needs (the server has no internet access at run time).
# RUN pip install --no-cache-dir <package>

WORKDIR /app
# Copies the code and model.pt; data, checkpoints and logs are excluded by .dockerignore.
COPY . /app

# For local tests: docker run --rm --gpus all -v <in>:/data -v <out>:/output my-submission
ENTRYPOINT ["python", "-m", "submission.run"]
