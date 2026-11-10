FROM python:3.11-slim

RUN python -m pip install --no-cache-dir pytest==8.3.5 \
    && groupadd --gid 10001 sandbox \
    && useradd --uid 10001 --gid 10001 --no-create-home sandbox

WORKDIR /workspace
USER 10001:10001
CMD ["python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
