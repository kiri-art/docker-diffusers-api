# Banana requires Cuda version 11+.  Below is banana default:
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel as base
# xformers available precompiled for:
#   Python 3.9 or 3.10, CUDA 11.3 or 11.6, and PyTorch 1.12.1
#   https://github.com/facebookresearch/xformers/#getting-started
# Below: pytorch base images only have Python 3.7 :(
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime as base
# Below: our ideal image, but Optimization fails with it.
#FROM continuumio/miniconda3:4.12.0 as base

# Note, docker uses HTTP_PROXY and HTTPS_PROXY (uppercase)
# We purposefully want those managed independently, as we want docker
# to manage its own cache.  This is just for pip, models, etc.
ARG http_proxy
ENV http_proxy=${http_proxy}
ARG https_proxy
ENV https_proxy=${https_proxy}
RUN if [ -n "$http_proxy" ] ; then \
    echo quit \
    | openssl s_client -proxy $(echo ${https_proxy} | cut -b 8-) -servername google.com -connect google.com:443 -showcerts \
    | sed 'H;1h;$!d;x; s/^.*\(-----BEGIN CERTIFICATE-----.*-----END CERTIFICATE-----\)\n---\nServer certificate.*$/\1/' \
    > /usr/local/share/ca-certificates/squid-self-signed.crt ; \
    update-ca-certificates ; \
  fi
ENV REQUESTS_CA_BUNDLE=${http_proxy:+/usr/local/share/ca-certificates/squid-self-signed.crt}

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

## XXXX playing around a lot.
# pip installs pytorch 1.13 and uninstalls 1.12 (needed by xformers)
# recomment conda update; didn't help.  need to solve above issue.

RUN conda update -n base -c defaults conda
# We need python 3.9 or 3.10 for xformers
# Yes, we install pytorch twice... will switch base image in future
RUN conda create -n xformers python=3.10
SHELL ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "xformers", "/bin/bash", "-c"]
RUN python --version
RUN conda install -c pytorch -c conda-forge cudatoolkit=11.6 pytorch=1.12.1
RUN conda install xformers -c xformers/label/dev

# Install python packages
# RUN pip3 install --upgrade pip
RUN https_proxy="" REQUESTS_CA_BUNDLE="" conda install pip
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Not needed anymore, but, may be needed again in the future :D
# Turing: 7.5 (RTX 20s, Quadro), Ampere: 8.0 (A100), 8.6 (RTX 30s)
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"

RUN git clone https://github.com/huggingface/diffusers
WORKDIR /api/diffusers
RUN git checkout v0.9.0
WORKDIR /api
RUN pip install -e diffusers

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
# "CompVis/stable-diffusion-v1-4", "hakurei/waifu-diffusion",
# "stabilityai/stable-diffusion-2",
# "stabilityai/stable-diffusion-2-inpainting" etc.
ARG MODEL_ID="stabilityai/stable-diffusion-2"
ENV MODEL_ID=${MODEL_ID}

# "" = model default.
ARG PRECISION="fp16"
ENV PRECISION=${PRECISION}
ADD precision.py .

# ARG PIPELINE="StableDiffusionInpaintPipeline"
ARG PIPELINE="ALL"
ENV PIPELINE=${PIPELINE}

ARG USE_DREAMBOOTH=0
ENV USE_DREAMBOOTH=${USE_DREAMBOOTH}

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
ARG AWS_S3_ENDPOINT_URL
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
ENV AWS_S3_ENDPOINT_URL=${AWS_S3_ENDPOINT_URL}

COPY utils utils

# Download diffusers model from somewhere else (see Storage docs)
# Don't use this for checkpoints (.ckpt)!  Use CHECKPOINT_URL for that.
ARG MODEL_URL=""
ENV MODEL_URL=${MODEL_URL}
# If set, it will be downloaded and converted to diffusers format, and
# saved in a directory with same MODEL_ID name to be loaded by diffusers.
ARG CHECKPOINT_URL=""
ENV CHECKPOINT_URL=${CHECKPOINT_URL}
ARG CHECKPOINT_CONFIG_URL=""
ENV CHECKPOINT_CONFIG_URL=${CHECKPOINT_CONFIG_URL}

ADD download-checkpoint.py .
RUN python3 download-checkpoint.py
ARG _CONVERT_SPECIAL
ENV _CONVERT_SPECIAL=${_CONVERT_SPECIAL}
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

RUN if [ "$USE_DREAMBOOTH" = "1" ] ; then \
    # By specifying the same torch version as conda, it won't download again.
    # Without this, it will upgrade torch, break xformers, make bigger image.
    pip install -r diffusers/examples/dreambooth/requirements.txt bitsandbytes torch==1.12.1 ; \
  fi
RUN if [ "$USE_DREAMBOOTH" = "1" ] ; then apt-get install git-lfs ; fi

# Add your custom app code, init() and inference()
ADD train_dreambooth.py .
ADD send.py .
ADD app.py .

ARG SEND_URL
ENV SEND_URL=${SEND_URL}
ARG SIGN_KEY
ENV SIGN_KEY=${SIGN_KEY}

CMD python3 -u server.py

