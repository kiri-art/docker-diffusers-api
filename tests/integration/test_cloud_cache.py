import sys
from .lib import getMinio, getDDA
from test import runTest

def test_cloud_cache_create_and_upload():
    """
    Check if model exists in cloud cache bucket download otherwise, save
    with safetensors, and upload model.tar.zst to bucket
    """
    minio = getMinio()
    print(minio)
    dda = getDDA(minio=minio)
    print(dda)

    ## bucket.objects.all().delete()
    result = runTest(
        "txt2img",
        { "test_url": dda["url"] },
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
            # "MODEL_ID": "hf-internal-testing/tiny-stable-diffusion-pipe",
            "MODEL_PRECISION": "fp16",
            "MODEL_REVISION": "fp16",
            "MODEL_URL": "s3://",
        },
        {"num_inference_steps": 1},
    )

    timings = result["$timings"]
    assert timings["download"] > 0
    assert timings["upload"] > 0
