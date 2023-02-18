import sys
from .lib import getMinio, getDDA
from test import runTest


def test_cloudcache_build_download():
    """
    Download a model from cloud-cache at build time (no HuggingFace)
    """
    minio = getMinio()
    print(minio)
    environment = {
        "RUNTIME_DOWNLOADS": 0,
        "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
        "MODEL_PRECISION": "fp16",
        "MODEL_REVISION": "fp16",
        "MODEL_URL": "s3://",  # <--
    }
    conda = "conda run --no-capture-output -n xformers"
    dda = getDDA(
        minio=minio,
        stream_logs=True,
        environment=environment,
        root_cache=False,
        command=[
            "sh",
            "-c",
            f"{conda} python3 -u download.py && ls -l && {conda} python3 -u server.py",
        ],
    )
    print(dda)
    assert dda.container.status == "running"

    ## bucket.objects.all().delete()
    result = runTest(
        "txt2img",
        {"test_url": dda.url},
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
        },
        {"num_inference_steps": 1},
    )

    dda.stop()
    minio.stop()
    assert result["image_base64"]
    print("test successs\n\n")


def test_huggingface_build_download():
    """
    Download a model from HuggingFace at build time (no cloud-cache)
    NOTE / TODO: Good starting point, but this still runs with gpu and
    uploads if missing.
    """
    environment = {
        "RUNTIME_DOWNLOADS": 0,
        "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
        "MODEL_PRECISION": "fp16",
        "MODEL_REVISION": "fp16",
    }
    conda = "conda run --no-capture-output -n xformers"
    dda = getDDA(
        stream_logs=True,
        environment=environment,
        root_cache=False,
        command=[
            "sh",
            "-c",
            f"{conda} python3 -u download.py && ls -l && {conda} python3 -u server.py",
        ],
    )
    print(dda)
    assert dda.container.status == "running"

    ## bucket.objects.all().delete()
    result = runTest(
        "txt2img",
        {"test_url": dda.url},
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
            # "MODEL_ID": "hf-internal-testing/tiny-stable-diffusion-pipe",
            "MODEL_PRECISION": "fp16",
            "MODEL_REVISION": "fp16",
            "MODEL_URL": "",  # <-- no model_url, i.e. no cloud cache
        },
        {"num_inference_steps": 1},
    )
    dda.stop()

    assert result["image_base64"]
    print("test successs\n\n")


def test_checkpoint_url_build_download():
    """
    Download and convert a .ckpt at build time.  No cloud-cache.
    """
    environment = {
        "RUNTIME_DOWNLOADS": 0,
        "MODEL_ID": "hakurei/waifu-diffusion-v1-3",
        "MODEL_PRECISION": "fp16",
        "MODEL_REVISION": "fp16",
        "CHECKPOINT_URL": "https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float16.ckpt",
    }
    conda = "conda run --no-capture-output -n xformers"
    dda = getDDA(
        stream_logs=True,
        environment=environment,
        root_cache=False,
        command=[
            "sh",
            "-c",
            f"{conda} python3 -u download.py && ls -l && {conda} python3 -u server.py",
        ],
    )
    print(dda)
    assert dda.container.status == "running"

    ## bucket.objects.all().delete()
    result = runTest(
        "txt2img",
        {"test_url": dda.url},
        {
            "MODEL_ID": "hakurei/waifu-diffusion-v1-3",
            "MODEL_PRECISION": "fp16",
            "MODEL_URL": "",  # <-- no model_url, i.e. no cloud cache
        },
        {"num_inference_steps": 1},
    )
    dda.stop()

    assert result["image_base64"]
    print("test successs\n\n")
