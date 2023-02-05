import torch
import os
from diffusers import pipelines as _pipelines, StableDiffusionPipeline
from getScheduler import getScheduler, DEFAULT_SCHEDULER
from precision import torch_dtype_from_precision
from device import device
import time

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
PIPELINE = os.getenv("PIPELINE")
USE_DREAMBOOTH = True if os.getenv("USE_DREAMBOOTH") == "1" else False
HOME = os.path.expanduser("~")
MODELS_DIR = os.path.join(HOME, ".cache", "diffusers-api")


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


def loadModel(model_id: str, load=True, precision=None, revision=None):
    torch_dtype = torch_dtype_from_precision(precision)
    if revision == "":
        revision = None

    print(
        "loadModel",
        {
            "model_id": model_id,
            "load": load,
            "precision": precision,
            "revision": revision,
        },
    )
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

    model_dir = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(model_dir):
        model_dir = None

    from_pretrained = time.time()
    model = pipeline.from_pretrained(
        model_dir or model_id,
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
        model.to(device)
        to_gpu = round((time.time() - to_gpu) * 1000)
        print(f"Loaded from disk in {from_pretrained} ms, to gpu in {to_gpu} ms")
    else:
        print(f"Downloaded in {from_pretrained} ms")

    return model if load else None
