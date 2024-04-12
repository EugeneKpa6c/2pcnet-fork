FROM ubuntu:20.04
#FROM python:3.11-slim

WORKDIR /app

# Устанавливаем необходимые пакеты и библиотеки
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update --fix-missing && apt-get install -y python3 python3-pip

# соращаем установку
ENV DEBIAN_FRONTEND=noninteractive



# добавляем все катологи для скачивания
COPY pip.conf pip.conf
ENV PIP_CONFIG_FILE pip.conf

RUN apt update
RUN apt install curl -y

# Устанавливаем необходимые пакеты и библиотеки
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

RUN apt-get update && apt-get install -y git
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
COPY . /app

# Запускаем оболочку bash при старте контейнера
CMD ["bash"]