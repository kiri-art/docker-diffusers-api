#!/bin/bash

DOCKER_GW_IP="172.17.0.1"

MINIO_CONTAINER_ID=""
DIFFUSERS_CONTAINER_ID=""
HF_AUTH_TOKEN="$HF_AUTH_TOKEN"
AWS_ACCESS_KEY_ID="minioadmin"
AWS_SECRET_ACCESS_KEY="minioadmin"
AWS_DEFAULT_REGION=""
AWS_S3_ENDPOINT_URL="http://$DOCKER_GW_IP:9010"
AWS_S3_DEFAULT_BUCKET="test"
TEST_URL=""

init() {
  if [ ! -f $HOME/minio-binaries/mc ]; then
    curl https://dl.min.io/client/mc/release/linux-amd64/mc \
      --create-dirs \
      -o $HOME/minio-binaries/mc

    chmod +x $HOME/minio-binaries/mc
  fi

  export PATH=$PATH:$HOME/minio-binaries/

  docker run -d --rm \
    --name minio-test -p 9010:9000 -p 9011:9001 \
    minio/minio server /data --console-address ":9011"

  MINIO_CONTAINER_ID=$(docker ps -qf name=minio-test)

  sleep 5
  mc alias set minio-test $AWS_S3_ENDPOINT_URL minioadmin minioadmin
  mc admin info minio-test
  mc mb minio-test/$AWS_S3_DEFAULT_BUCKET

  # Historically we used to build as part of the tests.
  # docker build -t diffusers-api-test .
  # docker build -t gadicc/diffusers-api .

  docker run -d --rm \
    --gpus all  \
    -p 8010:8000 \
    --name diffusers-api-test \
    -e HF_AUTH_TOKEN="$HF_AUTH_TOKEN" \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
    -e AWS_S3_ENDPOINT_URL="$AWS_S3_ENDPOINT_URL" \
    -e AWS_S3_DEFAULT_BUCKET="test" \
    "$@" gadicc/diffusers-api

#    -e http_proxy="http://172.17.0.1:3128" \
#    -e https_proxy="http://172.17.0.1:3128" \
#    -e REQUESTS_CA_BUNDLE="/usr/local/share/ca-certificates/squid-self-signed.crt" \

  DIFFUSERS_CONTAINER_ID=$(docker ps -qf name=diffusers-api-test)
  TEST_URL="http://localhost:8010/"

  # TODO, loop to figure out exactly when we're up
  sleep 5
  curl ${TEST_URL}healthcheck
}

shutdown() {
  # Containers were all started with --rm
  echo "Shutting down MINIO container $MINIO_CONTAINER_ID"
  docker stop $MINIO_CONTAINER_ID
  echo "Shutting down DIFFUSERS container $DIFFUSERS_CONTAINER_ID"
  docker stop $DIFFUSERS_CONTAINER_ID
}

# python tests_create_cache.py

init

export TEST_URL HF_AUTH_TOKEN
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION AWS_S3_ENDPOINT_URL AWS_S3_DEFAULT_BUCKET

python -m pytest -s ./tests_create_cache.py
RETURN_VALUE=$?
if [ $RETURN_VALUE = 0 ]; then
  echo "TESTS PASSED"
else
  echo "TESTS FAILED"
fi

shutdown
exit $RETURN_VALUE
