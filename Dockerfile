ARG FROM_IMAGE="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
# ARG FROM_IMAGE="gadicc/diffusers-api-base:python3.9-pytorch1.12.1-cuda11.6-xformers"
# You only need the -banana variant if you need banana's optimization
# i.e. not relevant if you're using RUNTIME_DOWNLOADS
# ARG FROM_IMAGE="gadicc/python3.9-pytorch1.12.1-cuda11.6-xformers-banana"
FROM ${FROM_IMAGE} as base
ENV FROM_IMAGE=${FROM_IMAGE}

# Note, docker uses HTTP_PROXY and HTTPS_PROXY (uppercase)
# We purposefully want those managed independently, as we want docker
# to manage its own cache.  This is just for pip, models, etc.
ARG http_proxy
ARG https_proxy
RUN if [ -n "$http_proxy" ] ; then \
    echo quit \
    | openssl s_client -proxy $(echo ${https_proxy} | cut -b 8-) -servername google.com -connect google.com:443 -showcerts \
    | sed 'H;1h;$!d;x; s/^.*\(-----BEGIN CERTIFICATE-----.*-----END CERTIFICATE-----\)\n---\nServer certificate.*$/\1/' \
    > /usr/local/share/ca-certificates/squid-self-signed.crt ; \
    update-ca-certificates ; \
  fi
ARG REQUESTS_CA_BUNDLE=${http_proxy:+/usr/local/share/ca-certificates/squid-self-signed.crt}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -yq apt-utils
RUN apt-get install -yqq git zstd wget curl

FROM base AS patchmatch
ARG USE_PATCHMATCH=0
WORKDIR /tmp
COPY scripts/patchmatch-setup.sh .
RUN sh patchmatch-setup.sh

FROM base as output
RUN mkdir /api
WORKDIR /api

# we use latest pip in base image
# RUN pip3 install --upgrade pip

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

# [b9feed8] move to 0.19.0dev (#4048)
ARG DIFFUSERS_VERSION="b9feed87958c27074b0618cc543696c05f58e2c9"
ENV DIFFUSERS_VERSION=${DIFFUSERS_VERSION}

RUN git clone https://github.com/huggingface/diffusers && cd diffusers && git checkout ${DIFFUSERS_VERSION}
WORKDIR /api
RUN pip install -e diffusers

# Set to true to NOT download model at build time, rather at init / usage.
ARG RUNTIME_DOWNLOADS=1
ENV RUNTIME_DOWNLOADS=${RUNTIME_DOWNLOADS}

# TODO, to dda-bananana
# ARG PIPELINE="StableDiffusionInpaintPipeline"
ARG PIPELINE="ALL"
ENV PIPELINE=${PIPELINE}

# Deps for RUNNING (not building) earlier options
ARG USE_PATCHMATCH=0
RUN if [ "$USE_PATCHMATCH" = "1" ] ; then apt-get install -yqq python3-opencv ; fi
COPY --from=patchmatch /tmp/PyPatchMatch PyPatchMatch

# TODO, just include by default, and handle all deps in OUR requirements.txt
ARG USE_DREAMBOOTH=1
ENV USE_DREAMBOOTH=${USE_DREAMBOOTH}

RUN if [ "$USE_DREAMBOOTH" = "1" ] ; then \
    # By specifying the same torch version as conda, it won't download again.
    # Without this, it will upgrade torch, break xformers, make bigger image.
    # bitsandbytes==0.40.0.post4 had failed cuda detection on dreambooth test.
    pip install -r diffusers/examples/dreambooth/requirements.txt ; \
  fi
RUN if [ "$USE_DREAMBOOTH" = "1" ] ; then apt-get install -yqq git-lfs ; fi

ARG USE_REALESRGAN=1
RUN if [ "$USE_REALESRGAN" = "1" ] ; then apt-get install -yqq libgl1-mesa-glx libglib2.0-0 ; fi
RUN if [ "$USE_REALESRGAN" = "1" ] ; then git clone https://github.com/xinntao/Real-ESRGAN.git ; fi
# RUN if [ "$USE_REALESRGAN" = "1" ] ; then pip install numba==0.57.1 chardet ; fi
RUN if [ "$USE_REALESRGAN" = "1" ] ; then pip install basicsr==1.4.2 facexlib==0.2.5 gfpgan==1.3.8 ; fi
RUN if [ "$USE_REALESRGAN" = "1" ] ; then cd Real-ESRGAN && python3 setup.py develop ; fi

COPY api/ .
EXPOSE 8000

ARG SAFETENSORS_FAST_GPU=1
ENV SAFETENSORS_FAST_GPU=${SAFETENSORS_FAST_GPU}

CMD python3 -u server.py

