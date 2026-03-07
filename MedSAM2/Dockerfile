FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

# Combine apt steps and cleanup in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends git python3-pip python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Combine clone + venv + installs in one layer to avoid caching intermediate states
RUN git clone https://github.com/jcoloradot-unal/Segmentacion-Modelo.git && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -e '/workspace/Segmentacion-Modelo/MedSAM2[dev]' && \
    /opt/venv/bin/pip install --no-cache-dir fastapi uvicorn

ENV PATH="/opt/venv/bin:$PATH"

COPY /checkpoints/MedSAM2_latest.pt /workspace/Segmentacion-Modelo/MedSAM2/checkpoints

EXPOSE 8001

CMD ["python3", "/workspace/Segmentacion-Modelo/MedSAM2/app/requests_handler.py"]