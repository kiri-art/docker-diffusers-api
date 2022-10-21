import torch
import os
from diffusers import StableDiffusionPipeline

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
MODEL_IDS = [
    "CompVis/stable-diffusion-v1-4",
    "hakurei/waifu-diffusion",
    "hakurei/waifu-diffusion-v1-3",
    "runwayml/stable-diffusion-inpainting",
    "runwayml/stable-diffusion-v1-5",
]


def loadModel(model_id: str, load=True):
    print(("Loading" if load else "Downloading") + " model: " + model_id)

    model = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=HF_AUTH_TOKEN,
    )

    return model.to("cuda") if load else None
