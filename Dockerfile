FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./ultralytics /app/ultralytics

RUN pip3 install --upgrade pip
RUN pip3 install -e /app/ultralytics
RUN pip3 install --ignore-installed blinker
RUN pip3 install mlflow boto3 jetson-stats fabric onnxslim onnxruntime

COPY src/ ./src/
ENTRYPOINT ["python3", "src/jetson_test.py"]
