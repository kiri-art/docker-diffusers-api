# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import StableDiffusionPipeline
import torch
import os
from DOWNLOAD_VARS import MODEL_ID

MODEL_IDS = ["CompVis/stable-diffusion-v1-4", "hakurei/waifu-diffusion"]


def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    # Set auth token which is required to download stable diffusion model weights
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    # Bad for production serverless, great for local dev & preview deploys
    if MODEL_ID == "ALL":
        for MODEL_I in MODEL_IDS:
            StableDiffusionPipeline.from_pretrained(
                MODEL_I,
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=HF_AUTH_TOKEN,
            )
    else:
        model = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=HF_AUTH_TOKEN,
        )


if __name__ == "__main__":
    download_model()
