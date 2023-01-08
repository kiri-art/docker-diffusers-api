import sys
import os
from .lib import getMinio, getDDA
from test import runTest


def test_memory():
    return
    """
    Make sure when switching models we release VRAM afterwards.
    """
    HOME = os.getenv("HOME")
    dda = dda = getDDA(
        # minio=minio
        # stream_logs=True,
        volumes=[f"{HOME}/root-cache:/root/.cache"],
    )
    print(dda)

    TEST_ARGS = {"test_url": dda.url}
    MODEL_ARGS = {"num_infernece_steps": 1}

    mem_usage = list()

    # 768x768 model is obviously bigger
    result = runTest(
        "txt2img",
        TEST_ARGS,
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1",  # not -base
            "MODEL_REVISION": "fp16",
            "MODEL_PRECISION": "fp16",
            "MODEL_URL": "s3://",
        },
        MODEL_ARGS,
    )
    print(result)
    mem_usage.append(result["$mem_usage"])

    # 512x512 model is obviously bigger
    result = runTest(
        "txt2img",
        TEST_ARGS,
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",  # <--
            "MODEL_REVISION": "fp16",
            "MODEL_PRECISION": "fp16",
            "MODEL_URL": "s3://",
        },
        MODEL_ARGS,
    )
    print(result)
    mem_usage.append(result["$mem_usage"])

    print({"mem_usage": mem_usage})
    assert mem_usage[1] < mem_usage[0]

    dda.stop()
