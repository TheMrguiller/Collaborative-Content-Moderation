FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.2-cuda12.1.0


RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/repo/src

COPY ./requirements.txt .
COPY ./credentials/kaggle.json ~/.kaggle/kaggle.json
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
EXPOSE 8888
COPY ./src .
WORKDIR /home/repo/



CMD ["bash"]