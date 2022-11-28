# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
import os
import json
import sys
import time
import datetime
import argparse
import distutils
from uuid import uuid4
from io import BytesIO
from PIL import Image
from pathlib import Path, PosixPath

path = os.path.dirname(os.path.realpath(sys.argv[0]))
TESTS = path + os.sep + "tests"
FIXTURES = TESTS + os.sep + "fixtures"
OUTPUT = TESTS + os.sep + "output"
Path(OUTPUT).mkdir(parents=True, exist_ok=True)


def b64encode_file(filename: str):
    path = (
        filename
        if isinstance(filename, PosixPath)
        else os.path.join(FIXTURES, filename)
    )
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("ascii")


def output_path(filename: str):
    return os.path.join(OUTPUT, filename)


def decode_and_save(image_byte_string: str, name: str):
    image_encoded = image_byte_string.encode("utf-8")
    image_bytes = BytesIO(base64.b64decode(image_encoded))
    image = Image.open(image_bytes)
    fp = output_path(name + ".png")
    image.save(fp)
    print("Saved " + fp)


all_tests = {}


def test(name, inputs):
    global all_tests
    all_tests.update({name: inputs})


def runTest(name, banana, extraCallInputs, extraModelInputs):
    inputs = all_tests.get(name)
    inputs.get("callInputs").update(extraCallInputs)
    inputs.get("modelInputs").update(extraModelInputs)

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
        callID = result.get("callID")

        if result.get("finished", None) == False:
            while result.get(
                "message", None
            ) != "success" and not "error" in result.get("message", None):
                secondsSinceStart = round((time.time() - start) / 1000)
                print(str(datetime.datetime.now()) + f": t+{secondsSinceStart}s")
                print(json.dumps(result, indent=4))
                print
                payload = {
                    "id": str(uuid4()),
                    "created": int(time.time()),
                    "longPoll": True,
                    "apiKey": BANANA_API_KEY,
                    "callID": callID,
                }
                response = requests.post(
                    "https://api.banana.dev/check/v4/", json=payload
                )
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
        timings_str = json.dumps(
            dict(
                map(
                    lambda item: (
                        item[0],
                        f"{item[1]/1000/60:.1f}m"
                        if item[1] > 60000
                        else f"{item[1]/1000:.1f}s"
                        if item[1] > 1000
                        else str(item[1]) + "ms",
                    ),
                    timings.items(),
                )
            )
        ).replace('"', "")[1:-1]
        print(f"Request took {finish:.1f}s ({timings_str})")
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
            "SCHEDULER": "LMSDiscreteScheduler",
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
            "SCHEDULER": "LMSDiscreteScheduler",
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
            "SCHEDULER": "LMSDiscreteScheduler",
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
            "SCHEDULER": "DDIMScheduler",  # Note, as of diffusers 0.3.0, no LMS yet
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
            "SCHEDULER": "DDIMScheduler",  # Note, as of diffusers 0.3.0, no LMS yet
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
                "SCHEDULER": "DDIMScheduler",  # Note, as of diffusers 0.3.0, no LMS yet
                "FILL_MODE": "patchmatch",
            },
        },
    )

# Actually we just want this to be a non-default test?
if True or os.getenv("USE_DREAMBOOTH"):
    test(
        "dreambooth",
        # If you're calling from the command line, don't forget to a
        # specify a destination if you want your fine-tuned model to
        # be uploaded somewhere at the end.
        {
            "modelInputs": {
                "instance_prompt": "a photo of sks dog",
                "instance_images": list(
                    map(
                        b64encode_file,
                        list(Path("tests/fixtures/dreambooth").iterdir()),
                    )
                ),
                # Option 1: upload to HuggingFace (see notes below)
                # Make sure your HF API token has read/write access.
                # "hub_model_id": "huggingFaceUsername/targetModelName",
                # "push_to_hub": True,
            },
            "callInputs": {
                "MODEL_ID": "runwayml/stable-diffusion-v1-5",
                "PIPELINE": "StableDiffusionPipeline",
                "SCHEDULER": "DDPMScheduler",
                "train": "dreambooth",
                # Option 2: store on S3.  Note the **s3:///* (x3).  See notes below.
                # "dest_url": "s3:///bucket/filename.tar.zst".
            },
        },
    )


def main(tests_to_run, banana, extraCallInputs, extraModelInputs):
    invalid_tests = []
    for test in tests_to_run:
        if all_tests.get(test, None) == None:
            invalid_tests.append(test)

    if len(invalid_tests) > 0:
        print("No such tests: " + ", ".join(invalid_tests))
        exit(1)

    for test in tests_to_run:
        runTest(test, banana, extraCallInputs, extraModelInputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--banana", required=False, action="store_true")
    parser.add_argument(
        "--xmfe",
        required=False,
        default=None,
        type=lambda x: bool(distutils.util.strtobool(x)),
    )
    parser.add_argument("--scheduler", required=False, type=str)
    parser.add_argument("--call-arg", action="append", type=str)
    parser.add_argument("--model-arg", action="append", type=str)

    args, tests_to_run = parser.parse_known_args()

    call_inputs = {}
    model_inputs = {}

    if args.call_arg:
        for arg in args.call_arg:
            name, value = arg.split("=", 1)
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)
            call_inputs.update({name: value})

    if args.model_arg:
        for arg in args.model_arg:
            name, value = arg.split("=", 1)
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)
            model_inputs.update({name: value})

    if args.xmfe != None:
        call_inputs.update({"xformers_memory_efficient_attention": args.xmfe})
    if args.scheduler:
        call_inputs.update({"SCHEDULER": args.scheduler})

    if len(tests_to_run) < 1:
        print(
            "Usage: python3 test.py [--banana] [--xmfe=1/0] [--scheduler=SomeScheduler] [all / test1] [test2] [etc]"
        )
        sys.exit()
    elif len(tests_to_run) == 1 and (
        tests_to_run[0] == "ALL" or tests_to_run[0] == "all"
    ):
        tests_to_run = list(all_tests.keys())

    main(
        tests_to_run,
        banana=args.banana,
        extraCallInputs=call_inputs,
        extraModelInputs=model_inputs,
    )
