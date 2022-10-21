from sched import scheduler
import torch

from torch import autocast
from diffusers import (
    pipelines as _pipelines,
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DiffusionPipeline,
)
import base64
from io import BytesIO
import PIL
import json
from loadModel import loadModel
from send import send
import os

MODEL_ID = os.environ.get("MODEL_ID")

PIPELINES = [
    "StableDiffusionPipeline",
    "StableDiffusionImg2ImgPipeline",
    "StableDiffusionInpaintPipeline",
]

SCHEDULERS = ["LMS", "DDIM", "PNDM"]

torch.set_grad_enabled(False)


def createPipelinesFromModel(model):
    pipelines = dict()
    for pipeline in PIPELINES:
        pipelines[pipeline] = getattr(_pipelines, pipeline)(**model.components)
    return pipelines


class DummySafetyChecker:
    @staticmethod
    def __call__(images, clip_input):
        return images, False


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model  # needed for bananna optimizations
    global pipelines
    global schedulers
    global dummy_safety_checker

    send(
        "init",
        "start",
        {
            "device": torch.cuda.get_device_name(),
            "hostname": os.getenv("HOSTNAME"),
            "model_id": MODEL_ID,
        },
        True,
    )

    schedulers = {
        "LMS": LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        ),
        "DDIM": DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        ),
        "PNDM": PNDMScheduler(),
    }

    dummy_safety_checker = DummySafetyChecker()

    if MODEL_ID == "ALL":
        global last_model_id
        last_model_id = None
        return

    model = loadModel(MODEL_ID)

    pipelines = createPipelinesFromModel(model)

    send("init", "done")


def decodeBase64Image(imageStr: str, name: str) -> PIL.Image:
    image = PIL.Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))
    print(f'Decoded image "{name}": {image.format} {image.width}x{image.height}')
    return image


def truncateInputs(inputs: dict):
    clone = inputs.copy()
    if "modelInputs" in clone:
        modelInputs = clone["modelInputs"] = clone["modelInputs"].copy()
        for item in ["init_image", "mask_image", "image"]:
            if item in modelInputs:
                modelInputs[item] = modelInputs[item][0:6] + "..."
    return clone


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(all_inputs: dict) -> dict:
    global model
    global pipelines
    global last_model_id
    global schedulers
    global dummy_safety_checker

    print(json.dumps(truncateInputs(all_inputs), indent=2))
    model_inputs = all_inputs.get("modelInputs", None)
    call_inputs = all_inputs.get("callInputs", None)
    startRequestId = call_inputs.get("startRequestId", None)

    # Fallback until all clients on new code
    if model_inputs == None:
        return {"$error": "UPGRADE CLIENT - no model_inputs specified"}

    model_id = call_inputs.get("MODEL_ID")
    if MODEL_ID == "ALL":
        if last_model_id != model_id:
            model = loadModel(model_id)
            pipelines = createPipelinesFromModel(model)
            last_model_id = model_id
    else:
        if model_id != MODEL_ID:
            return {
                "$error": {
                    "code": "MODEL_MISMATCH",
                    "message": f'Model "{model_id}" not available on this container which hosts "{MODEL_ID}"',
                    "requested": model_id,
                    "available": MODEL_ID,
                }
            }

    pipeline = pipelines.get(call_inputs.get("PIPELINE"))

    pipeline.scheduler = schedulers.get(call_inputs.get("SCHEDULER"))

    safety_checker = call_inputs.get("safety_checker", True)
    pipeline.safety_checker = (
        model.safety_checker if safety_checker else dummy_safety_checker
    )

    # Parse out your arguments
    # prompt = model_inputs.get("prompt", None)
    # if prompt == None:
    #     return {"message": "No prompt provided"}
    #
    #   height = model_inputs.get("height", 512)
    #  width = model_inputs.get("width", 512)
    # num_inference_steps = model_inputs.get("num_inference_steps", 50)
    # guidance_scale = model_inputs.get("guidance_scale", 7.5)
    # seed = model_inputs.get("seed", None)
    #   strength = model_inputs.get("strength", 0.75)

    if "init_image" in model_inputs:
        model_inputs["init_image"] = decodeBase64Image(
            model_inputs.get("init_image"), "init_image"
        )

    if "image" in model_inputs:
        model_inputs["image"] = decodeBase64Image(model_inputs.get("image"), "image")

    if "mask_image" in model_inputs:
        model_inputs["mask_image"] = decodeBase64Image(
            model_inputs.get("mask_image"), "mask_image"
        )

    seed = model_inputs.get("seed", None)
    if seed == None:
        generator = torch.Generator(device="cuda")
        generator.seed()
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        del model_inputs["seed"]

    model_inputs.update({"generator": generator})

    send("inference", "start", {"startRequestId": startRequestId}, True)

    # Run the model
    # with autocast("cuda"):
    # image = pipeline(**model_inputs).images[0]

    with torch.inference_mode():
        # autocast im2img and inpaint which are broken in 0.4.0, 0.4.1
        # still broken in 0.5.1
        if call_inputs.get("PIPELINE") != "StableDiffusionPipeline":
            with autocast("cuda"):
                images = pipeline(**model_inputs).images
        else:
            images = pipeline(**model_inputs).images

    images_base64 = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        images_base64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    send("inference", "done", {"startRequestId": startRequestId})

    # Return the results as a dictionary
    if len(images_base64) > 1:
        return {"images_base64": images_base64}

    return {"image_base64": images_base64[0]}
