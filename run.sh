#!/bin/bash

docker run -it --rm \
  --gpus all  \
  -p 8000:8000 \
  -e http_proxy="http://172.17.0.1:3128" \
  -e https_proxy="http://172.17.0.1:3128" \
  -e REQUESTS_CA_BUNDLE="/usr/local/share/ca-certificates/squid-self-signed.crt" \
  -e HF_AUTH_TOKEN="$HF_AUTH_TOKEN" \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
  -e AWS_S3_ENDPOINT_URL="$AWS_S3_ENDPOINT_URL" \
  -e AWS_S3_DEFAULT_BUCKET="$AWS_S3_DEFAULT_BUCKET" \
  -v ~/root-cache:/root/.cache \
  "$@" gadicc/diffusers-api
