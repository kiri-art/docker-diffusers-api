# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
import os
import json
import sys
import time
import argparse
import distutils
from uuid import uuid4
from io import BytesIO
from PIL import Image
from pathlib import Path

path = os.path.dirname(os.path.realpath(sys.argv[0]))
TESTS = path + os.sep + "tests"
FIXTURES = TESTS + os.sep + "fixtures"
OUTPUT = TESTS + os.sep + "output"
Path(OUTPUT).mkdir(parents=True, exist_ok=True)


def b64encode_file(filename: str):
    with open(os.path.join(FIXTURES, filename), "rb") as file:
        return base64.b64encode(file.read())


def output_path(filename: str):
    return os.path.join(OUTPUT, filename)


def decode_and_save(image_byte_string: str, name: str):
    image_encoded = image_byte_string.encode("utf-8")
    image_bytes = BytesIO(base64.b64decode(image_encoded))
    image = Image.open(image_bytes)
    fp = output_path(name + ".png")
    image.save(fp)
    print("Saved " + fp)


tests = {}


def test(name, inputs):
    global tests
    tests.update({name: inputs})


def runTest(name, banana, extraCallInputs):
    inputs = tests.get(name)
    inputs.get("callInputs").update(extraCallInputs)

    print("Running test: " + name)

    start = time.time()
    if banana:
        BANANA_API_KEY = os.getenv("BANANA_API_KEY")
        BANANA_MODEL_KEY = os.getenv("BANANA_MODEL_KEY")
        if BANANA_MODEL_KEY == None or BANANA_API_KEY == None:
            print("Error: BANANA_API_KEY or BANANA_MODEL_KEY not set, aborting...")
            sys.exit(1)

        payload = {
            "id": str(uuid4()),
            "created": int(time.time()),
            "apiKey": BANANA_API_KEY,
            "modelKey": BANANA_MODEL_KEY,
            "modelInputs": inputs,
            "startOnly": False,
        }
        response = requests.post("https://api.banana.dev/start/v4/", json=payload)

        result = response.json()
        modelOutputs = result.get("modelOutputs", None)
        if modelOutputs == None:
            finish = time.time() - start
            print(f"Request took {finish:.1f}s")
            print(result)
            return
        result = modelOutputs[0]
    else:
        response = requests.post("http://localhost:8000/", json=inputs)
        result = response.json()

    finish = time.time() - start
    timings = result.get("$timings")
    if timings:
        init = timings.get("init") / 1000
        inference = timings.get("inference") / 1000
        print(
            f"Request took {finish:.1f}s ("
            + f"init: {init:.1f}s, inference: {inference:.1f}s)"
        )
    else:
        print(f"Request took {finish:.1f}s")

    if (
        result.get("images_base64", None) == None
        and result.get("image_base64", None) == None
    ):
        print(json.dumps(result, indent=4))
        print()
        return

    images_base64 = result.get("images_base64", None)
    if images_base64:
        for idx, image_byte_string in enumerate(images_base64):
            decode_and_save(image_byte_string, f"{name}_{idx}")
    else:
        decode_and_save(result["image_base64"], name)

    print()


test(
    "txt2img",
    {
        "modelInputs": {"prompt": "realistic field of grass"},
        "callInputs": {
            "MODEL_ID": "runwayml/stable-diffusion-v1-5",
            "PIPELINE": "StableDiffusionPipeline",
            "SCHEDULER": "LMS",
            # "xformers_memory_efficient_attention": False,
        },
    },
)

# multiple images
test(
    "txt2img-multiple",
    {
        "modelInputs": {
            "prompt": "realistic field of grass",
            "num_images_per_prompt": 2,
        },
        "callInputs": {
            "MODEL_ID": "runwayml/stable-diffusion-v1-5",
            "PIPELINE": "StableDiffusionPipeline",
            "SCHEDULER": "LMS",
        },
    },
)


test(
    "img2img",
    {
        "modelInputs": {
            "prompt": "A fantasy landscape, trending on artstation",
            "init_image": b64encode_file("sketch-mountains-input.jpg"),
        },
        "callInputs": {
            "MODEL_ID": "runwayml/stable-diffusion-v1-5",
            "PIPELINE": "StableDiffusionImg2ImgPipeline",
            "SCHEDULER": "LMS",
        },
    },
)

test(
    "inpaint-v1-4",
    {
        "modelInputs": {
            "prompt": "a cat sitting on a bench",
            "init_image": b64encode_file("overture-creations-5sI6fQgYIuo.png"),
            "mask_image": b64encode_file("overture-creations-5sI6fQgYIuo_mask.png"),
        },
        "callInputs": {
            "MODEL_ID": "CompVis/stable-diffusion-v1-4",
            "PIPELINE": "StableDiffusionInpaintPipelineLegacy",
            "SCHEDULER": "DDIM",  # Note, as of diffusers 0.3.0, no LMS yet
        },
    },
)

test(
    "inpaint-sd",
    {
        "modelInputs": {
            "prompt": "a cat sitting on a bench",
            "image": b64encode_file("overture-creations-5sI6fQgYIuo.png"),
            "mask_image": b64encode_file("overture-creations-5sI6fQgYIuo_mask.png"),
        },
        "callInputs": {
            "MODEL_ID": "runwayml/stable-diffusion-inpainting",
            "PIPELINE": "StableDiffusionInpaintPipeline",
            "SCHEDULER": "DDIM",  # Note, as of diffusers 0.3.0, no LMS yet
        },
    },
)

if os.getenv("USE_PATCHMATCH"):
    test(
        "outpaint",
        {
            "modelInputs": {
                "prompt": "girl with a pearl earing standing in a big room",
                "init_image": b64encode_file(
                    "girl_with_pearl_earing_outpainting_in.png"
                ),
            },
            "callInputs": {
                "MODEL_ID": "CompVis/stable-diffusion-v1-4",
                "PIPELINE": "StableDiffusionInpaintPipelineLegacy",
                "SCHEDULER": "DDIM",  # Note, as of diffusers 0.3.0, no LMS yet
                "FILL_MODE": "patchmatch",
            },
        },
    )


def main(tests_to_run, banana, extraCallInputs):
    invalid_tests = []
    for test in tests_to_run:
        if tests.get(test, None) == None:
            invalid_tests.append(test)

    if len(invalid_tests) > 0:
        print("No such tests: " + ", ".join(invalid_tests))
        exit(1)

    for test in tests_to_run:
        runTest(test, banana, extraCallInputs)


if __name__ == "__main__":
    print(
        "Usage: python3 test.py [--banana] [--xmfe=1/0] [--scheduler=SomeScheduler] [test1] [test2] [etc]"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--banana", required=False, action="store_true")
    parser.add_argument(
        "--xmfe",
        required=False,
        default=True,
        type=lambda x: bool(distutils.util.strtobool(x)),
    )
    parser.add_argument("--scheduler", required=False, type=str)

    args, tests_to_run = parser.parse_known_args()
    extraCallInputs = {"xformers_memory_efficient_attention": args.xmfe}
    if args.scheduler:
        extraCallInputs.update({"SCHEDULER": args.scheduler})

    main(tests_to_run, banana=args.banana, extraCallInputs=extraCallInputs)
