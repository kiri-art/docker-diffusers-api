import pytest
import docker
import atexit
import time
import boto3
import os
import requests
import socket
import asyncio
import sys
import subprocess
import selectors
from threading import Thread
from argparse import Namespace

AWS_S3_DEFAULT_BUCKET = os.environ.get("AWS_S3_DEFAULT_BUCKET", "test")
DOCKER_GW_IP = "172.17.0.1"  # will override below if found

myContainers = list()
dockerClient = docker.DockerClient(
    base_url="unix://var/run/docker.sock", version="auto"
)
for network in dockerClient.networks.list():
    if network.attrs["Scope"] == "local" and network.attrs["Driver"] == "bridge":
        DOCKER_GW_IP = network.attrs["IPAM"]["Config"][0]["Gateway"]
        break

# # https://stackoverflow.com/a/53255955/1839099
# def fire_and_forget(f):
#     def wrapped(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(None, f, *args, *kwargs)
#     return wrapped
#
# @fire_and_forget
# def log_streamer(container):
#   for line in container.logs(stream=True):
#     print(line.decode(), end="")


def log_streamer(container, name=None):
    """
    Streams logs to stdout/stderr.
    Order is not guaranteed (have tried 3 different methods)
    """
    # Method 1: pipe streams directly -- even this doesn't guarantee order
    # Method 2: threads + readline
    # Method 3: selectors + read1
    method = 1

    if method == 1:
        kwargs = {
            "stdout": sys.stdout,
            "stderr": sys.stderr,
        }
    elif method == 2:
        kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 1,
            "universal_newlines": True,
        }
    elif method == 3:
        kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 1,
        }

    prefix = f"[{name or container.id[:7]}] "
    print(prefix + "== Streaming logs (stdout/stderr order not guaranteed): ==")

    sp = subprocess.Popen(["docker", "logs", "-f", container.id], **kwargs)

    if method == 2:

        def reader(pipe):
            while True:
                read = pipe.readline()
                if read == "" and sp.poll() is not None:
                    break
                print(prefix + read, end="")
                sys.stdout.flush()
                sys.stderr.flush()

        Thread(target=reader, args=[sp.stdout]).start()
        Thread(target=reader, args=[sp.stderr]).start()

    elif method == 3:
        selector = selectors.DefaultSelector()
        selector.register(sp.stdout, selectors.EVENT_READ)
        selector.register(sp.stderr, selectors.EVENT_READ)
        loop = True

        while loop:
            for key, _ in selector.select():
                data = key.fileobj.read1().decode()
                if not data:
                    loop = False
                    break
                line = prefix + str(data).rstrip().replace("\n", "\n" + prefix)
                if key.fileobj is sp.stdout:
                    print(line)
                    sys.stdout.flush()
                else:
                    print(line, file=sys.stderr)
                    sys.stderr.flush()


def get_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def startContainer(image, command=None, stream_logs=False, onstop=None, **kwargs):
    global myContainers

    container = dockerClient.containers.run(
        image,
        command,
        # auto_remove=True,
        detach=True,
        **kwargs,
    )

    if stream_logs:
        log_streamer(container)

    myContainers.append(container)

    def stop():
        print("stop", container.id)
        container.stop()
        container.remove()
        myContainers.remove(container)
        if onstop:
            onstop()

    while container.status != "running" and container.status != "exited":
        time.sleep(1)
        try:
            container.reload()
        except Exception as error:
            print(container.logs())
            raise error
        print(container.status)

    # if (container.status == "exited"):
    #  print(container.logs())
    #  raise Exception("unexpected exit")

    print("returned", container)
    return container, stop


_minioCache = {}


def getMinio(id="disposable"):
    cached = _minioCache.get(id, None)
    if cached:
        return Namespace(**cached)

    if id == "global":
        endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL")
        if endpoint_url:
            print("Reusing existing global minio")
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_s3_default_bucket = AWS_S3_DEFAULT_BUCKET
            s3 = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                config=boto3.session.Config(signature_version="s3v4"),
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=None,
                # verify=False,
            )
            result = {
                # don't link to actual container, and don't rm it at end
                "container": "global",
                "stop": lambda: print(),
                # "port": port,
                "endpoint_url": endpoint_url,
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
                "aws_s3_default_bucket": aws_s3_default_bucket,
                "s3": s3,
            }
            _minioCache.update({id: result})
            return Namespace(**result)
        else:
            print("Creating new global minio")

    port = get_free_port()

    def onstop():
        del _minioCache[id]

    container, stop = startContainer(
        "minio/minio",
        "server /data --console-address :9001",
        ports={9000: port},
        onstop=onstop,
    )

    endpoint_url = f"http://{DOCKER_GW_IP}:{port}"

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

    aws_access_key_id = "minioadmin"
    aws_secret_access_key = "minioadmin"
    aws_s3_default_bucket = AWS_S3_DEFAULT_BUCKET
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        config=boto3.session.Config(signature_version="s3v4"),
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=None,
        # verify=False,
    )

    s3.create_bucket(Bucket=AWS_S3_DEFAULT_BUCKET)

    result = {
        "container": container,
        "stop": stop,
        "port": port,
        "endpoint_url": endpoint_url,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_s3_default_bucket": aws_s3_default_bucket,
        "s3": s3,
    }
    _minioCache.update({id: result})
    return Namespace(**result)


_ddaCache = None


def getDDA(
    minio=None,
    command=None,
    environment={},
    stream_logs=False,
    wait=True,
    root_cache=True,
    **kwargs,
):
    global _ddaCache
    if _ddaCache:
        print("return _ddaCache")
        return Namespace(**_ddaCache)
    else:
        print("create new _dda")

    port = get_free_port()

    environment.update(
        {
            "HF_AUTH_TOKEN": os.getenv("HF_AUTH_TOKEN"),
            "http_proxy": os.getenv("DDA_http_proxy"),
            "https_proxy": os.getenv("DDA_https_proxy"),
            "REQUESTS_CA_BUNDLE": os.getenv("DDA_http_proxy")
            and "/usr/local/share/ca-certificates/squid-self-signed.crt",
        }
    )

    if minio:
        environment.update(
            {
                "AWS_ACCESS_KEY_ID": minio.aws_access_key_id,
                "AWS_SECRET_ACCESS_KEY": minio.aws_secret_access_key,
                "AWS_DEFAULT_REGION": "",
                "AWS_S3_DEFAULT_BUCKET": minio.aws_s3_default_bucket,
                "AWS_S3_ENDPOINT_URL": minio.endpoint_url,
            }
        )

    def onstop():
        global _ddaCache
        _ddaCache = None

    HOME = os.getenv("HOME")

    container, stop = startContainer(
        "gadicc/diffusers-api:test",
        command,
        stream_logs=stream_logs,
        ports={8000: port},
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        environment=environment,
        volumes=root_cache and [f"{HOME}/root-cache:/root/.cache"],
        onstop=onstop,
        **kwargs,
    )

    url = f"http://{DOCKER_GW_IP}:{port}/"

    while wait:
        time.sleep(1)
        container.reload()
        if container.status == "exited":
            if not stream_logs:
                print("--- EARLY EXIT ---")
                print(container.logs().decode())
                print("--- EARLY EXIT ---")
            raise Exception("Early exit before successful healthcheck")

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
                if result["state"] == "healthy" and result["gpu"] == True:
                    print("Ready")
                    break
                else:
                    print(response)
                    print(response.text)
            else:
                raise Exception("Unexpected status code from dda/healthcheck")

    data = {
        "container": container,
        "stop": stop,
        "minio": minio,
        "port": port,
        "url": url,
    }

    _ddaCache = data
    return Namespace(**data)


def cleanup():
    print("cleanup")
    for container in myContainers:
        print("Stopping")
        print(container)
        container.stop()
        print("removing")
        container.remove()


atexit.register(cleanup)
