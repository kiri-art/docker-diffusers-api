# Banana requires Cuda version 11+.  Below is banana default:
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel as base
# xformers available precompiled for:
#   Python 3.9 or 3.10, CUDA 11.3 or 11.6, and PyTorch 1.12.1
#   https://github.com/facebookresearch/xformers/#getting-started
# Below: pytorch base images only have Python 3.7 :(
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime as base
# Below: our ideal image, but Optimization fails with it.
#FROM continuumio/miniconda3:4.12.0 as base
ENV DEBIAN_FRONTEND=noninteractive
#RUN apt-get install gnupg2
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -yqq git

# This would have been great but Python is via conda,
# and conda doesn't support python >= 3.7 for base.
#RUN apt install -yqq software-properties-common
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt update
#RUN apt-get install -yqq python3.10
#RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
#RUN ln -sf /usr/bin/python3.10 /usr/bin/python

FROM base AS patchmatch
ARG USE_PATCHMATCH=0
WORKDIR /tmp
COPY scripts/patchmatch-setup.sh .
RUN sh patchmatch-setup.sh

FROM base as output
RUN mkdir /api
WORKDIR /api

# We need python 3.9 or 3.10 for xformers
# Yes, we install pytorch twice... will switch base image in future
# RUN conda update -n base -c defaults conda
RUN conda create -n xformers python=3.10
SHELL ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "xformers", "/bin/bash", "-c"]
RUN python --version
RUN conda install -c pytorch -c conda-forge cudatoolkit=11.6 pytorch=1.12.1
RUN conda install xformers -c xformers/label/dev

# Install python packages
RUN mkdir -p /root/.cache/pip
COPY root-cache/pip /root/.cache/pip
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Required to build flash attention
# Turing: 7.5 (RTX 20s, Quadro), Ampere: 8.0 (A100), 8.6 (RTX 30s)
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# ENV FLASH_ATTENTION=0
# ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"
# this is built it into memory efficient attention now ! ^_^

ADD install.sh .
RUN bash install.sh

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Dev: docker build --build-arg HF_AUTH_TOKEN=${HF_AUTH_TOKEN} ...
# Banana: currently, comment out ARG and set by hand ENV line.
ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=${HF_AUTH_TOKEN}

# MODEL_ID, can be any of:
# 1) Hugging face model name
# 2) A directory containing a diffusers model
# 3) Your own unique model id if using CHECKPOINT_URL below.
# 4) "ALL" to download all known models (useful for dev)
# "runwayml/stable-diffusion-v1-5", "runwayml/stable-diffusion-inpainting"
# "CompVis/stable-diffusion-v1-4", "hakurei/waifu-diffusion","stabilityai/stable-diffusion-2",stabilityai/stable-diffusion-2-inpainting etc.
ARG MODEL_ID="runwayml/stable-diffusion-v1-5"
ENV MODEL_ID=${MODEL_ID}

# ARG PIPELINE="StableDiffusionInpaintPipeline"
ARG PIPELINE="ALL"
ENV PIPELINE=${PIPELINE}

COPY root-cache/huggingface /root/.cache/huggingface
COPY root-cache/checkpoints /root/.cache/checkpoints
RUN du -sh /root/.cache/*

# If set, it will be downloaded and converted to diffusers format, and
# saved in a directory with same MODEL_ID name to be loaded by diffusers.
ARG CHECKPOINT_URL=""
ENV CHECKPOINT_URL=${CHECKPOINT_URL}
ADD download-checkpoint.py .
RUN python3 download-checkpoint.py
ADD convert-to-diffusers.py .
RUN python3 convert-to-diffusers.py
# RUN rm -rf checkpoints

# Add your model weight files 
# (in this case we have a python script)
ADD getScheduler.py .
ADD loadModel.py .
ADD download.py .
RUN python3 download.py

# Deps for RUNNING (not building) earlier options
ARG USE_PATCHMATCH=0
RUN if [ "$USE_PATCHMATCH" = "1" ] ; then apt-get install -yqq python3-opencv ; fi
COPY --from=patchmatch /tmp/PyPatchMatch PyPatchMatch

# Add your custom app code, init() and inference()
ADD send.py .
ADD app.py .

CMD python3 -u server.py

