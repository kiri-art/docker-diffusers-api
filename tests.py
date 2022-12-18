from test import runTest


def main():
    result = runTest(
        "txt2img",
        False,
        {"MODEL_ID": "CompVis/stable-diffusion-v1-4", "MODEL_URL": "s3://"},
        {"num_inference_steps": 1},
    )
    mem_usage = list()
    mem_usage.append(result["$mem_usage"])
    result = runTest(
        "txt2img",
        False,
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
            "MODEL_PRECISION": "fp16",
            "MODEL_URL": "s3://",
        },
        {"num_inference_steps": 1},
    )
    mem_usage.append(result["$mem_usage"])
    result = runTest(
        "txt2img",
        False,
        {"MODEL_ID": "CompVis/stable-diffusion-v1-4", "MODEL_URL": "s3://"},
        {"num_inference_steps": 1},
    )
    mem_usage.append(result["$mem_usage"])
    print({"mem_usage": mem_usage})


if __name__ == "__main__":
    main()
