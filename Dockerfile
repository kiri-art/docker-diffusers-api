# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel as base
ENV DEBIAN_FRONTEND=noninteractive
RUN mkdir -p /root/.cache/pip
COPY root-cache/pip /root/.cache/pip
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -yqq git

FROM base AS patchmatch
ARG USE_PATCHMATCH=0
WORKDIR /tmp
COPY scripts/patchmatch-setup.sh .
RUN sh patchmatch-setup.sh

FROM base as output
RUN mkdir /api
WORKDIR /api

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Required to build flash attention
# Turing: 7.5 (RTX 20s, Quadro), Ampere: 8.0 (A100), 8.6 (RTX 30s)
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
ENV FLASH_ATTENTION=0
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"

ADD install.sh .
RUN bash install.sh

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Dev: docker build --build-arg HF_AUTH_TOKEN=${HF_AUTH_TOKEN} ...
# Banana: currently, comment out ARG and set by hand ENV line.
ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=${HF_AUTH_TOKEN}

# MODEL_ID
# 1) Hugging face model name
# 2) A directory containing a diffusers model
# 3) Your own unique model id if using CHECKPOINT_URL below.
# 4) "ALL" to download all known models (useful for dev)
# "runwayml/stable-diffusion-v1-5", "runwayml/stable-diffusion-inpainting"
# "CompVis/stable-diffusion-v1-4", "hakurei/waifu-diffusion", etc.
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
