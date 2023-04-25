ARG FROM_IMAGE="gadicc/diffusers-api-base:python3.9-pytorch1.12.1-cuda11.6-xformers"
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

# [9965cb5] [Community Pipelines] Update lpw_stable_diffusion pipeline (#3197)
# Above was reverted shortly afterwards for not being backwards compatible.
RUN git clone https://github.com/huggingface/diffusers && cd diffusers && git checkout 9965cb50eac12e397473f01535aab43aae76b4ab
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
    pip install -r diffusers/examples/dreambooth/requirements.txt bitsandbytes torch==1.12.1 ; \
  fi
RUN if [ "$USE_DREAMBOOTH" = "1" ] ; then apt-get install git-lfs ; fi

COPY api/ .
EXPOSE 8000

ARG SAFETENSORS_FAST_GPU=1
ENV SAFETENSORS_FAST_GPU=${SAFETENSORS_FAST_GPU}

CMD python3 -u server.py

