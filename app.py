from sched import scheduler
import torch
from torch import autocast
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
)
import base64
from io import BytesIO
import os
from APP_VARS import MODEL_ID, PIPELINE, SCHEDULER

PIPELINES = [
    "StableDiffusionPipeline",
    "StableDiffusionImg2ImgPipeline",
    "StableDiffusionInpaintPipeline",
]


def getPipeline(PIPELINE):
    print("PIPELINE = '" + PIPELINE + "'")
    if PIPELINE == "StableDiffusionPipeline":
        return StableDiffusionPipeline
    if PIPELINE == "StableDiffusionImg2ImgPipeline":
        return StableDiffusionImg2ImgPipeline
    if PIPELINE == "StableDiffusionInpaintPipeline":
        return StableDiffusionInpaintPipeline


def initScheduler(SCHEDULER):
    print("SCHEDULER = '" + SCHEDULER + "'")
    if SCHEDULER == "LMS":
        return LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
    elif SCHEDULER == "DDIM":
        return DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
    else:
        return PNDMScheduler()


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    if MODEL_ID == "ALL":
        global last_model_id, last_pipeline_name, last_scheduler_name
        last_model_id = None
        last_pipeline_name = None
        last_scheduler_name = None
        return

    print("MODEL_ID = '" + MODEL_ID + "'")
    pipeline = getPipeline(PIPELINE)
    scheduler = initScheduler(SCHEDULER)

    model = pipeline.from_pretrained(
        MODEL_ID,
        revision="fp16",
        torch_dtype=torch.float16,
        scheduler=scheduler,
        use_auth_token=HF_AUTH_TOKEN,
    ).to("cuda")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model
    global last_model_id, last_pipeline_name, last_scheduler_name

    if MODEL_ID == "ALL":
        HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

        model_id = model_inputs.get("MODEL_ID")
        pipeline_name = model_inputs.get("PIPELINE")
        scheduler_name = model_inputs.get("SCHEDULER")

        if (
            last_model_id != model_id
            or last_pipeline_name != pipeline_name
            or last_scheduler_name != scheduler_name
        ):
            print("Loading model...")
            print("MODEL_ID = '" + model_id + "'")
            pipeline = getPipeline(pipeline_name)
            scheduler = initScheduler(scheduler_name)
            model = pipeline.from_pretrained(
                model_id,
                revision="fp16",
                torch_dtype=torch.float16,
                scheduler=scheduler,
                use_auth_token=HF_AUTH_TOKEN,
            ).to("cuda")
            last_model_id = model_id
            last_pipeline_name = pipeline_name
            last_scheduler_name = scheduler_name

    # Parse out your arguments
    prompt = model_inputs.get("prompt", None)
    if prompt == None:
        return {"message": "No prompt provided"}

    height = model_inputs.get("height", 512)
    width = model_inputs.get("width", 512)
    num_inference_steps = model_inputs.get("num_inference_steps", 50)
    guidance_scale = model_inputs.get("guidance_scale", 7.5)
    seed = model_inputs.get("seed", None)

    if seed == None:
        # generator = None;
        generator = torch.Generator(device="cuda")
        generator.seed()
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Run the model
    with autocast("cuda"):
        image = model(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return the results as a dictionary
    return {"image_base64": image_base64}
