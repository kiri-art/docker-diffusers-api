import pytest
import docker
import atexit
import time
import boto3
import os
import requests

AWS_S3_DEFAULT_BUCKET="test"
DOCKER_GW_IP="172.17.0.1" # will override below if found

myContainers = list()
dockerClient = docker.DockerClient(base_url='unix://var/run/docker.sock', version="auto")
for network in dockerClient.networks.list():
  if (network.attrs["Scope"] == "local" and network.attrs["Driver"] == "bridge"):
    DOCKER_GW_IP=network.attrs["IPAM"]["Config"][0]["Gateway"]
    break

i = 0

def startContainer(image, command = None, **kwargs):
  container = dockerClient.containers.run(
    image,
    command,
    auto_remove=True,
    detach=True,
    **kwargs
  )
  myContainers.append(container)

  while container.status != "running":
    time.sleep(1)
    container.reload()
    print(container.status)
  return container

_minioCache = None
def getMinio():
  global _minioCache
  if _minioCache:
    return _minioCache

  container = startContainer(
    "minio/minio",
    "server /data --console-address :9011",
    ports={9000:9010,9011:9011},
  )

  endpoint_url = f"http://{DOCKER_GW_IP}:9010"

  while True:
    time.sleep(1)
    response = None
    try:
      print(endpoint_url + "/minio/health/live")
      response = requests.get(endpoint_url + "/minio/health/live")
    except Exception as error:
      print(error)

    if response and response.status_code == 200:
      break

  s3 = boto3.client(
    's3',
    endpoint_url=endpoint_url,
    config=boto3.session.Config(signature_version='s3v4'),
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    aws_session_token=None,
    # verify=False,
  )

  s3.create_bucket(Bucket=AWS_S3_DEFAULT_BUCKET)

  result = {
    "container": container,
    "endpoint_url": endpoint_url,
    "s3": s3,
  }
  _minioCache = result
  return result

_ddaCache = None
def getDDA(minio = None):
  global _ddaCache
  if _ddaCache:
    return _ddaCache

  PORT=8010

  environment = {
      "HF_AUTH_TOKEN": os.getenv("HF_AUTH_TOKEN"),
      "http_proxy": os.getenv("DDA_http_proxy"),
      "https_proxy": os.getenv("DDA_https_proxy"),
      "REQUESTS_CA_BUNDLE": os.getenv("DDA_http_proxy") and "/usr/local/share/ca-certificates/squid-self-signed.crt"
  }

  if minio:
    environment.update({
      "AWS_ACCESS_KEY_ID": "minioadmin",
      "AWS_SECRET_ACCESS_KEY": "minioadmin",
      "AWS_DEFAULT_REGION": "",
      "AWS_S3_DEFAULT_BUCKET": "test",
      "AWS_S3_ENDPOINT_URL": minio["endpoint_url"],
    })

  container = startContainer(
    "gadicc/diffusers-api",
    ports={8000:PORT},
    device_requests=[
        docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
    ],
    environment=environment,
  )

  url = f"http://{DOCKER_GW_IP}:{PORT}/"

  while True:
    time.sleep(1)
    response = None
    try:
      # print(url + "healthcheck")
      response = requests.get(url + "healthcheck")
    except Exception as error:
      # print(error)
      continue

    if response:
      if response.status_code == 200:
        result = response.json()
        if (result["state"] == "healthy" and result["gpu"] == True):
          print("Ready")
          break
        else:
          print(response)
          print(response.text)
      else:
        raise Exception("Unexpected status code from dda/healthcheck")

  data = {
    "container": container,
    "minio": minio,
    "url": url,
  }

  _ddaCache = data
  return data
  
def cleanup():
  print("cleanup")
  for container in myContainers:
    print("Stopping")
    print(container)
    container.stop()

atexit.register(cleanup)