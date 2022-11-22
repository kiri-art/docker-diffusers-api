from sched import scheduler
import torch

from torch import autocast
from diffusers import (
    pipelines as _pipelines,
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DiffusionPipeline,
    __version__,
)
import base64
from io import BytesIO
import PIL
import json
from loadModel import loadModel
from send import send, get_now
import os
import numpy as np
import skimage
import skimage.measure
from PyPatchMatch import patch_match
from getScheduler import getScheduler, SCHEDULERS
import re

USE_DREAMBOOTH = os.getenv("USE_DREAMBOOTH") == "1"
if USE_DREAMBOOTH:
    from train_dreambooth import TrainDreamBooth

MODEL_ID = os.environ.get("MODEL_ID")
PIPELINE = os.environ.get("PIPELINE")
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

PIPELINES = [
    "StableDiffusionPipeline",
    "StableDiffusionImg2ImgPipeline",
    "StableDiffusionInpaintPipeline",
    "StableDiffusionInpaintPipelineLegacy",
]

torch.set_grad_enabled(False)


def createPipelinesFromModel(model):
    pipelines = dict()
    for pipeline in PIPELINES:
        if hasattr(_pipelines, pipeline):
            if hasattr(model, "components"):
                pipelines[pipeline] = getattr(_pipelines, pipeline)(**model.components)
            else:
                pipelines[pipeline] = getattr(_pipelines, pipeline)(
                    vae=model.vae,
                    text_encoder=model.text_encoder,
                    tokenizer=model.tokenizer,
                    unet=model.unet,
                    scheduler=model.scheduler,
                    safety_checker=model.safety_checker,
                    feature_extractor=model.feature_extractor,
                )
        else:
            print(f'Skipping non-existent pipeline "{PIPELINE}"')
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
    global initTime

    initStart = get_now()
    send(
        "init",
        "start",
        {
            "device": torch.cuda.get_device_name(),
            "hostname": os.getenv("HOSTNAME"),
            "model_id": MODEL_ID,
            "diffusers": __version__,
        },
        True,
    )

    dummy_safety_checker = DummySafetyChecker()

    if MODEL_ID == "ALL":
        global last_model_id
        last_model_id = None
        return

    model = loadModel(MODEL_ID)

    if PIPELINE == "ALL":
        pipelines = createPipelinesFromModel(model)

    send("init", "done")
    initTime = get_now() - initStart


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
        if "instance_images" in modelInputs:
            modelInputs["instance_images"] = list(
                map(lambda str: str[0:6] + "...", modelInputs["instance_images"])
            )
    return clone


last_xformers_memory_efficient_attention = {}

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(all_inputs: dict) -> dict:
    global model
    global pipelines
    global last_model_id
    global schedulers
    global dummy_safety_checker
    global last_xformers_memory_efficient_attention

    print(json.dumps(truncateInputs(all_inputs), indent=2))
    model_inputs = all_inputs.get("modelInputs", None)
    call_inputs = all_inputs.get("callInputs", None)

    if model_inputs == None or call_inputs == None:
        return {
            "$error": {
                "code": "INVALID_INPUTS",
                "message": "Expecting on object like { modelInputs: {}, callInputs: {} } but got "
                + json.dumps(all_inputs),
            }
        }

    startRequestId = call_inputs.get("startRequestId", None)

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

    if PIPELINE == "ALL":
        pipeline = pipelines.get(call_inputs.get("PIPELINE"))
    else:
        pipeline = model

    pipeline.scheduler = getScheduler(MODEL_ID, call_inputs.get("SCHEDULER", None))
    if pipeline.scheduler == None:
        return {
            "$error": {
                "code": "INVALID_SCHEDULER",
                "message": "",
                "requeted": call_inputs.get("SCHEDULER", None),
                "available": ", ".join(SCHEDULERS),
            }
        }

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

    if "instance_images" in model_inputs:
        model_inputs["instance_images"] = list(
            map(
                lambda str: decodeBase64Image(str, "instance_image"),
                model_inputs["instance_images"],
            )
        )

    inferenceStart = get_now()
    send("inference", "start", {"startRequestId": startRequestId}, True)

    # Run patchmatch for inpainting
    if call_inputs.get("FILL_MODE", None) == "patchmatch":
        sel_buffer = np.array(model_inputs.get("init_image"))
        img = sel_buffer[:, :, 0:3]
        mask = sel_buffer[:, :, -1]
        img = patch_match.inpaint(img, mask=255 - mask, patch_size=3)
        model_inputs["init_image"] = PIL.Image.fromarray(img)
        mask = 255 - mask
        mask = skimage.measure.block_reduce(mask, (8, 8), np.max)
        mask = mask.repeat(8, axis=0).repeat(8, axis=1)
        model_inputs["mask_image"] = PIL.Image.fromarray(mask)

    # Turning on takes 3ms and turning off 1ms... don't worry, I've got your back :)
    x_m_e_a = call_inputs.get("xformers_memory_efficient_attention", True)
    last_x_m_e_a = last_xformers_memory_efficient_attention.get(pipeline, None)
    if x_m_e_a != last_x_m_e_a:
        if x_m_e_a == True:
            print("pipeline.enable_xformers_memory_efficient_attention()")
            pipeline.enable_xformers_memory_efficient_attention()  # default on
        elif x_m_e_a == False:
            print("pipeline.disable_xformers_memory_efficient_attention()")
            pipeline.disable_xformers_memory_efficient_attention()
        else:
            return {
                "$error": {
                    "code": "INVALID_XFORMERS_MEMORY_EFFICIENT_ATTENTION_VALUE",
                    "message": f'Model "{model_id}" not available on this container which hosts "{MODEL_ID}"',
                    "requested": x_m_e_a,
                    "available": [True, False],
                }
            }
        last_xformers_memory_efficient_attention.update({pipeline: x_m_e_a})

    # Run the model
    # with autocast("cuda"):
    # image = pipeline(**model_inputs).images[0]

    if call_inputs.get("train", None) == "dreambooth":
        if not USE_DREAMBOOTH:
            return {
                "$error": {
                    "code": "TRAIN_DREAMBOOTH_NOT_AVAILABLE",
                    "message": 'Called with callInput { train: "dreambooth" } but built with USE_DREAMBOOTH=0',
                }
            }
        torch.set_grad_enabled(True)
        result = TrainDreamBooth(model_id, pipeline, model_inputs, call_inputs)
        torch.set_grad_enabled(False)
        send("inference", "done", {"startRequestId": startRequestId})
        inferenceTime = get_now() - inferenceStart
        timings = result.get("$timings", {})
        timings = {"init": initTime, "inference": inferenceTime, **timings}
        result.update({"$timings": timings})
        return result

    # Do this after dreambooth as dreambooth accepts a seed int directly.
    seed = model_inputs.get("seed", None)
    if seed == None:
        generator = torch.Generator(device="cuda")
        generator.seed()
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        del model_inputs["seed"]

    model_inputs.update({"generator": generator})

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
    inferenceTime = get_now() - inferenceStart
    timings = {"init": initTime, "inference": inferenceTime}

    # Return the results as a dictionary
    if len(images_base64) > 1:
        return {"images_base64": images_base64, "$timings": timings}

    return {"image_base64": images_base64[0], "$timings": timings}
