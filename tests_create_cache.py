import pytest
import boto3
import os
import sys
from test import runTest
from botocore.client import Config

AWS_S3_ENDPOINT_URL = os.environ.get("AWS_S3_ENDPOINT_URL", None)
AWS_S3_DEFAULT_BUCKET = os.environ.get("AWS_S3_DEFAULT_BUCKET", None)

if AWS_S3_DEFAULT_BUCKET != "test":
    sys.stderr.write("Set AWS_S3_DEFAULT_BUCKET=test for tests")
    sys.exit(1)

s3 = boto3.resource(
    "s3",
    endpoint_url=AWS_S3_ENDPOINT_URL,
    config=Config(signature_version="s3v4"),
)
bucket = s3.Bucket(AWS_S3_DEFAULT_BUCKET)


def test_cloud_cache_create_and_upload():
    """
    Check if model exists in cloud cache bucket download otherwise, save
    with safetensors, and upload model.tar.zst to bucket
    """
    ## bucket.objects.all().delete()
    result = runTest(
        "txt2img",
        {},
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
            # "MODEL_ID": "hf-internal-testing/tiny-stable-diffusion-pipe",
            "MODEL_PRECISION": "fp16",
            "MODEL_URL": "s3://",
        },
        {"num_inference_steps": 1},
    )

    timings = result["$timings"]
    assert timings["download"] > 0
    assert timings["upload"] > 0
