import torch
import os
from diffusers import pipelines as _pipelines, StableDiffusionPipeline
from getScheduler import getScheduler, DEFAULT_SCHEDULER
from precision import revision, torch_dtype

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
PIPELINE = os.getenv("PIPELINE")
USE_DREAMBOOTH = True if os.getenv("USE_DREAMBOOTH") == "1" else False

MODEL_IDS = [
    "CompVis/stable-diffusion-v1-4",
    "hakurei/waifu-diffusion",
    # "hakurei/waifu-diffusion-v1-3", - not as diffusers yet
    "runwayml/stable-diffusion-inpainting",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2"
    "stabilityai/stable-diffusion-2-base"
    "stabilityai/stable-diffusion-2-inpainting",
]


def loadModel(model_id: str, load=True):
    print(("Loading" if load else "Downloading") + " model: " + model_id)

    pipeline = (
        StableDiffusionPipeline if PIPELINE == "ALL" else getattr(_pipelines, PIPELINE)
    )

    scheduler = getScheduler(model_id, DEFAULT_SCHEDULER, not load)

    model = pipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        use_auth_token=HF_AUTH_TOKEN,
        scheduler=scheduler,
        local_files_only=load,
        # Work around https://github.com/huggingface/diffusers/issues/1246
        # low_cpu_mem_usage=False if USE_DREAMBOOTH else True,
    )

    return model.to("cuda") if load else None
