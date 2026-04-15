FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed blinker
RUN pip3 install ultralytics mlflow boto3 jetson-stats fabric

COPY src/ ./src/
ENTRYPOINT ["python3", "src/test_in_jetson.py"]
