from test import runTest


def test_memory_free_on_swap_model():
    """
    Make sure memory is freed when swapping models at runtime.
    """
    result = runTest(
        "txt2img",
        {},
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
            "MODEL_PRECISION": "",  # full precision
            "MODEL_URL": "s3://",
        },
        {"num_inference_steps": 1},
    )
    mem_usage = list()
    mem_usage.append(result["$mem_usage"])
    result = runTest(
        "txt2img",
        {},
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
            "MODEL_PRECISION": "fp16",  # half precision
            "MODEL_URL": "s3://",
        },
        {"num_inference_steps": 1},
    )
    mem_usage.append(result["$mem_usage"])

    print({"mem_usage": mem_usage})
    # Assert that less memory used when unloading fp32 model and
    # loading the fp16 variant in its place
    assert mem_usage[1] < mem_usage[0]
