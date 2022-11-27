# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import os
from loadModel import loadModel, MODEL_IDS
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from precision import revision
from utils import Storage
import subprocess

MODEL_ID = os.environ.get("MODEL_ID")
MODEL_URL = os.environ.get("MODEL_URL")
USE_DREAMBOOTH = os.environ.get("USE_DREAMBOOTH")
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")


def download_model():
    if MODEL_URL != "":
        filename = MODEL_URL.split("/").pop()
        storage = Storage(MODEL_URL)
        storage.download_file(filename)
        os.mkdir(MODEL_ID)
        subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-C", MODEL_ID, "-xvf", filename],
            check=True,
        )
        subprocess.run(["ls", "-l"])
        return

    # do a dry run of loading the huggingface model, which will download weights at build time
    # For local dev & preview deploys, download all the models (terrible for serverless deploys)
    if MODEL_ID == "ALL":
        for MODEL_I in MODEL_IDS:
            loadModel(MODEL_I, False)
    else:
        loadModel(MODEL_ID, False)

    # if USE_DREAMBOOTH:
    # Actually we can re-use these from the above loaded model
    # Will remove this soon if no more surprises
    # for subfolder, model in [
    #     ["tokenizer", CLIPTokenizer],
    #     ["text_encoder", CLIPTextModel],
    #     ["vae", AutoencoderKL],
    #     ["unet", UNet2DConditionModel],
    #     ["scheduler", DDPMScheduler]
    # ]:
    #     print(subfolder, model)
    #     model.from_pretrained(
    #         MODEL_ID,
    #         subfolder=subfolder,
    #         revision=revision,
    #         use_auth_token=HF_AUTH_TOKEN,
    #     )


if __name__ == "__main__":
    download_model()
