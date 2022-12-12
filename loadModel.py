import torch
import os
from diffusers import pipelines as _pipelines, StableDiffusionPipeline
from getScheduler import getScheduler, DEFAULT_SCHEDULER
from precision import revision_from_precision, torch_dtype_from_precision
import time

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


def loadModel(model_id: str, load=True, precision=None):
    print("loadModel", {"model_id": model_id, "load": load, "precision": precision})
    revision = revision_from_precision(precision)
    torch_dtype = torch_dtype_from_precision(precision)
    print(
        ("Loading" if load else "Downloading")
        + " model: "
        + model_id
        + (f" ({revision})" if revision else "")
    )

    pipeline = (
        StableDiffusionPipeline if PIPELINE == "ALL" else getattr(_pipelines, PIPELINE)
    )

    scheduler = getScheduler(model_id, DEFAULT_SCHEDULER, not load)

    from_pretrained = time.time()
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
    from_pretrained = round((time.time() - from_pretrained) * 1000)

    if load:
        to_gpu = time.time()
        model.to("cuda")
        to_gpu = round((time.time() - to_gpu) * 1000)
        print(f"Loaded from disk in {from_pretrained} ms, to gpu in {to_gpu} ms")
    else:
        print(f"Downloaded in {from_pretrained} ms")

    return model if load else None
