# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import os
from loadModel import loadModel, MODEL_IDS
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from precision import revision
from utils import Storage
import subprocess
from pathlib import Path
from send import send as _send
import shutil

MODEL_ID = os.environ.get("MODEL_ID")
MODEL_URL = os.environ.get("MODEL_URL")
USE_DREAMBOOTH = os.environ.get("USE_DREAMBOOTH")
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
RUNTIME_DOWNLOADS = os.environ.get("RUNTIME_DOWNLOADS")

# i.e. don't run during build
def send(type: str, status: str, payload: dict = {}):
    if RUNTIME_DOWNLOADS:
        _send(type, status, payload)


def download_model(model_url=None, model_id=None):
    id = model_id or MODEL_ID
    url = model_url or MODEL_URL

    if url != "":
        filename = url.split("/").pop()
        storage = Storage(url)
        exists = storage.file_exists()
        if exists:
            storage.download_file(filename)
            # os.mkdir(id)
            Path(id).mkdir(parents=True, exist_ok=False)
            subprocess.run(
                ["tar", "--use-compress-program=unzstd", "-C", id, "-xvf", filename],
                check=True,
            )
            subprocess.run(["ls", "-l"])
        else:
            print("Does not exist, let's try find it on huggingface")
            download_model(model_id=model_id)
            model = loadModel(model_id, True)
            dir = model_id + "_save"
            model.save_pretrained(dir, safe_serialization=True)

            # This is all duped from train_dreambooth, need to refactor TODO XXX
            send("compress", "start", {})
            subprocess.run(
                f"tar cvf - -C {dir} . | zstd -o {filename}",
                shell=True,
                check=True,  # TODO, rather don't raise and return an error in JSON
            )

            send("compress", "done")
            subprocess.run(["ls", "-l", filename])

            send("upload", "start", {})
            upload_result = storage.upload_file(filename, filename)
            send("upload", "done")
            print(upload_result)
            os.remove(filename)
            shutil.rmtree(dir)
            # TODO, swap directories, inside HF's cache structure.

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
