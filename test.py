# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
import os
from io import BytesIO
from PIL import Image

TESTS = "tests"
FIXTURES = TESTS + os.sep + "fixtures"
OUTPUT = TESTS + os.sep + "output"


def b64encode_file(filename: str):
    with open(os.path.join(FIXTURES, filename), "rb") as file:
        return base64.b64encode(file.read())


def output_path(filename: str):
    return os.path.join(OUTPUT, filename)


def test(name, json):
    print("Running test: " + name)
    res = requests.post("http://localhost:8000/", json=json)

    image_byte_string = res.json()["image_base64"]

    image_encoded = image_byte_string.encode("utf-8")
    image_bytes = BytesIO(base64.b64decode(image_encoded))
    image = Image.open(image_bytes)
    fp = output_path(name + ".jpg")
    image.save(fp)
    print("Saved " + fp)
    print()


test(
    "txt2img",
    {
        "modelInputs": {"prompt": "realistic field of grass"},
        "callInputs": {
            "MODEL_ID": "CompVis/stable-diffusion-v1-4",
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
            "MODEL_ID": "CompVis/stable-diffusion-v1-4",
            "PIPELINE": "StableDiffusionImg2ImgPipeline",
            "SCHEDULER": "LMS",
        },
    },
)

test(
    "inpaint",
    {
        "modelInputs": {
            "prompt": "a cat sitting on a bench",
            "init_image": b64encode_file("overture-creations-5sI6fQgYIuo.png"),
            "mask_image": b64encode_file("overture-creations-5sI6fQgYIuo_mask.png"),
        },
        "callInputs": {
            "MODEL_ID": "CompVis/stable-diffusion-v1-4",
            "PIPELINE": "StableDiffusionInpaintPipeline",
            "SCHEDULER": "DDIM",  # Note, as of diffusers 0.3.0, no LMS yet
        },
    },
)
