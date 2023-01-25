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

# path = os.path.dirname(os.path.realpath(sys.argv[0]))
path = "."
TESTS = path + os.sep + "tests"
FIXTURES = TESTS + os.sep + "fixtures"
OUTPUT = TESTS + os.sep + "output"
TEST_URL = os.environ.get("TEST_URL", "http://localhost:8000/")
BANANA_API_URL = os.environ.get("BANANA_API_URL", "https://api.banana.dev")
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


# https://stackoverflow.com/a/1094933/1839099
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def decode_and_save(image_byte_string: str, name: str):
    image_encoded = image_byte_string.encode("utf-8")
    image_bytes = BytesIO(base64.b64decode(image_encoded))
    image = Image.open(image_bytes)
    fp = output_path(name + ".png")
    image.save(fp)
    print("Saved " + fp)
    size_formatted = sizeof_fmt(os.path.getsize(fp))

    return (
        f"[{image.width}x{image.height} {image.format} image, {size_formatted} bytes]"
    )


all_tests = {}


def test(name, inputs):
    global all_tests
    all_tests.update({name: inputs})


def runTest(name, args, extraCallInputs, extraModelInputs):
    origInputs = all_tests.get(name)
    inputs = {
        "modelInputs": origInputs.get("modelInputs", {}).copy(),
        "callInputs": origInputs.get("callInputs", {}).copy(),
    }
    inputs.get("callInputs").update(extraCallInputs)
    inputs.get("modelInputs").update(extraModelInputs)

    print("Running test: " + name)

    inputs_to_log = {
        "modelInputs": inputs["modelInputs"].copy(),
        "callInputs": inputs["callInputs"].copy(),
    }
    model_inputs_to_log = inputs_to_log["modelInputs"]

    for key in ["init_image", "image"]:
        if key in model_inputs_to_log:
            model_inputs_to_log[key] = "[image]"

    instance_images = model_inputs_to_log.get("instance_images", None)
    if instance_images:
        model_inputs_to_log["instance_images"] = f"[Array({len(instance_images)})]"

    print(json.dumps(inputs_to_log, indent=4))
    print()

    start = time.time()
    if args.get("banana", None):
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
        response = requests.post(f"{BANANA_API_URL}/start/v4/", json=payload)
        result = response.json()
        callID = result.get("callID")

        if result.get("finished", None) == False:
            while result.get(
                "message", None
            ) != "success" and not "error" in result.get("message", None):
                secondsSinceStart = time.time() - start
                print(str(datetime.datetime.now()) + f": t+{secondsSinceStart:.1f}s")
                print(json.dumps(result, indent=4))
                print
                payload = {
                    "id": str(uuid4()),
                    "created": int(time.time()),
                    "longPoll": True,
                    "apiKey": BANANA_API_KEY,
                    "callID": callID,
                }
                response = requests.post(f"{BANANA_API_URL}/check/v4/", json=payload)
                result = response.json()

        modelOutputs = result.get("modelOutputs", None)
        if modelOutputs == None:
            finish = time.time() - start
            print(f"Request took {finish:.1f}s")
            print(result)
            return
        result = modelOutputs[0]
    elif args.get("runpod", None):
        RUNPOD_API_URL = "https://api.runpod.ai/v1/"
        RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
        RUNPOD_MODEL_KEY = os.getenv("RUNPOD_MODEL_KEY")
        if not (RUNPOD_API_KEY and RUNPOD_MODEL_KEY):
            print("Error: RUNPOD_API_KEY or RUNPOD_MODEL_KEY not set, aborting...")
            sys.exit(1)

        url_base = RUNPOD_API_URL + RUNPOD_MODEL_KEY

        payload = {
            "input": inputs,
        }
        print(url_base + "/run")
        response = requests.post(
            url_base + "/run",
            json=payload,
            headers={"Authorization": "Bearer " + RUNPOD_API_KEY},
        )

        if response.status_code != 200:
            print("Unexpected HTTP response code: " + str(response.status_code))
            sys.exit(1)

        print(response)
        result = response.json()
        print(result)

        id = result["id"]

        while result["status"] != "COMPLETED":
            time.sleep(1)
            response = requests.get(
                f"{url_base}/status/{id}",
                headers={"Authorization": "Bearer " + RUNPOD_API_KEY},
            )
            result = response.json()

        result = result["output"]

    else:
        test_url = args.get("test_url", None) or TEST_URL
        response = requests.post(test_url, json=inputs)
        try:
            result = response.json()
        except requests.exceptions.JSONDecodeError as error:
            print(error)
            print(response.text)
            sys.exit(1)

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
        error = result.get("$error", None)
        if error:
            code = error.get("code", None)
            name = error.get("name", None)
            message = error.get("message", None)
            stack = error.get("stack", None)
            if code and name and message and stack:
                print()
                title = f"Exception {code} on container:"
                print(title)
                print("-" * len(title))
                # print(f'{name}("{message}")') - stack includes it.
                print(stack)
                return

        print(json.dumps(result, indent=4))
        print()
        return result

    images_base64 = result.get("images_base64", None)
    if images_base64:
        for idx, image_byte_string in enumerate(images_base64):
            images_base64[idx] = decode_and_save(image_byte_string, f"{name}_{idx}")
    else:
        result["image_base64"] = decode_and_save(result["image_base64"], name)

    print()
    print(json.dumps(result, indent=4))
    print()
    return result


test(
    "txt2img",
    {
        "modelInputs": {
            "prompt": "realistic field of grass",
            "num_inference_steps": 20,
        },
        "callInputs": {
            # "MODEL_ID": "<override_default>",  # (default)
            # "PIPELINE": "StableDiffusionPipeline",  # (default)
            # "SCHEDULER": "DPMSolverMultistepScheduler",  # (default)
            # "xformers_memory_efficient_attention": False,  # (default)
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
        }
    },
)


test(
    "img2img",
    {
        "modelInputs": {
            "prompt": "A fantasy landscape, trending on artstation",
            "image": b64encode_file("sketch-mountains-input.jpg"),
        },
        "callInputs": {
            "PIPELINE": "StableDiffusionImg2ImgPipeline",
        },
    },
)

test(
    "inpaint-v1-4",
    {
        "modelInputs": {
            "prompt": "a cat sitting on a bench",
            "image": b64encode_file("overture-creations-5sI6fQgYIuo.png"),
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

test(
    "checkpoint",
    {
        "modelInputs": {
            "prompt": "1girl",
        },
        "callInputs": {
            "MODEL_ID": "hakurei/waifu-diffusion-v1-3",
            "MODEL_URL": "s3://",
            "CHECKPOINT_URL": "http://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float16.ckpt",
        },
    },
)

if os.getenv("USE_PATCHMATCH"):
    test(
        "outpaint",
        {
            "modelInputs": {
                "prompt": "girl with a pearl earing standing in a big room",
                "image": b64encode_file("girl_with_pearl_earing_outpainting_in.png"),
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
                "train": "dreambooth",
                # Option 2: store on S3.  Note the **s3:///* (x3).  See notes below.
                # "dest_url": "s3:///bucket/filename.tar.zst".
            },
        },
    )


def main(tests_to_run, args, extraCallInputs, extraModelInputs):
    invalid_tests = []
    for test in tests_to_run:
        if all_tests.get(test, None) == None:
            invalid_tests.append(test)

    if len(invalid_tests) > 0:
        print("No such tests: " + ", ".join(invalid_tests))
        exit(1)

    for test in tests_to_run:
        runTest(test, args, extraCallInputs, extraModelInputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--banana", required=False, action="store_true")
    parser.add_argument("--runpod", required=False, action="store_true")
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
        vars(args),
        extraCallInputs=call_inputs,
        extraModelInputs=model_inputs,
    )
