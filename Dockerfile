# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN mkdir /api
WORKDIR /api

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Install git
RUN apt-get update && apt-get install -y git

COPY root-cache/* /root/.cache/

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Required to build flash attention
# Turing: 7.5 (RTX 20s, Quadro), Ampere: 8.0 (A100), 8.6 (RTX 30s)
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
ENV FLASH_ATTENTION=1
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"

ADD scripts/install.sh .
RUN bash install.sh

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Dev: docker build --build-arg HF_AUTH_TOKEN=${HF_AUTH_TOKEN} ...
# Banana: currently, comment out ARG and set by hand ENV line.
ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=${HF_AUTH_TOKEN}

# Which model to download and use; fork / downstream specific.
ADD DOWNLOAD_VARS.py .

# Add your model weight files 
# (in this case we have a python script)
ADD loadModel.py .
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD send.py .
ADD app.py .

# Runtime vars (for init and inference); fork / downstream specific.
ADD APP_VARS.py .

CMD python3 -u server.py
