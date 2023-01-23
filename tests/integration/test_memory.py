import sys
import os
from .lib import getMinio, getDDA
from test import runTest


def test_memory():
    """
    Make sure when switching models we release VRAM afterwards.
    """
    minio = getMinio("global")
    dda = getDDA(
        minio=minio,
        stream_logs=True,
    )
    print(dda)

    TEST_ARGS = {"test_url": dda.url}
    MODEL_ARGS = {"num_inference_steps": 1}

    mem_usage = list()

    # fp32 model is obviously bigger
    result = runTest(
        "txt2img",
        TEST_ARGS,
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
            "MODEL_REVISION": "",  # <--
            "MODEL_PRECISION": "",  # <--
            "MODEL_URL": "s3://",
        },
        MODEL_ARGS,
    )
    mem_usage.append(result["$mem_usage"])

    # fp32 model is obviously smaller
    result = runTest(
        "txt2img",
        TEST_ARGS,
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
            "MODEL_REVISION": "fp16",  # <--
            "MODEL_PRECISION": "fp16",  # <--
            "MODEL_URL": "s3://",
        },
        MODEL_ARGS,
    )
    mem_usage.append(result["$mem_usage"])

    print({"mem_usage": mem_usage})
    assert mem_usage[1] < mem_usage[0]

    dda.stop()
