# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import os
from loadModel import loadModel, MODEL_IDS
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from precision import PRECISION, revision_from_precision, torch_dtype_from_precision
from utils import Storage
import subprocess
from pathlib import Path
import shutil
from convert_to_diffusers import main as convert_to_diffusers
from download_checkpoint import main as download_checkpoint

MODEL_ID = os.environ.get("MODEL_ID")
MODEL_URL = os.environ.get("MODEL_URL")
USE_DREAMBOOTH = os.environ.get("USE_DREAMBOOTH")
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
RUNTIME_DOWNLOADS = os.environ.get("RUNTIME_DOWNLOADS")
HOME = os.path.expanduser("~")
MODELS_DIR = os.path.join(HOME, ".cache", "diffusers-api")
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

# i.e. don't run during build
def send(type: str, status: str, payload: dict = {}):
    if RUNTIME_DOWNLOADS:
        from send import send as _send

        _send(type, status, payload)


def normalize_model_id(model_id: str, model_revision):
    normalized_model_id = "models--" + model_id.replace("/", "--")
    if model_revision:
        normalized_model_id += "--" + model_revision
    return normalized_model_id


def download_model(
    model_url=None,
    model_id=None,
    model_revision=None,
    checkpoint_url=None,
    checkpoint_config_url=None,
):
    print(
        "download_model",
        {
            "model_url": model_url,
            "model_id": model_id,
            "model_revision": model_revision,
        },
    )
    id = model_id or MODEL_ID
    url = model_url or MODEL_URL
    revision = model_revision or revision_from_precision()
    normalized_model_id = id

    if url != "":
        normalized_model_id = normalize_model_id(model_id, model_revision)
        print({"normalized_model_id": normalized_model_id})
        filename = url.split("/").pop()
        if not filename:
            filename = normalized_model_id + ".tar.zst"
        model_file = os.path.join(MODELS_DIR, filename)
        storage = Storage(url, default_path=normalized_model_id + ".tar.zst")
        exists = storage.file_exists()
        if exists:
            storage.download_file(model_file)
            # os.mkdir(id)
            # Path(id).mkdir(parents=True, exist_ok=False)
            model_dir = os.path.join(MODELS_DIR, normalized_model_id)
            os.mkdir(model_dir)
            subprocess.run(
                [
                    "tar",
                    "--use-compress-program=unzstd",
                    "-C",
                    model_dir,
                    "-xvf",
                    model_file,
                ],
                check=True,
            )
            subprocess.run(["ls", "-l"])
            os.remove(model_file)
        else:
            if checkpoint_url:
                download_checkpoint(checkpoint_url)
                convert_to_diffusers(
                    model_id=model_id,
                    checkpoint_url=checkpoint_url,
                    checkpoint_config_url=checkpoint_config_url,
                )
            else:
                print("Does not exist, let's try find it on huggingface")
                print("precision = ", {"model_revision": model_revision})
                # This would be quicker to just model.to("cuda") afterwards, but
                # this conveniently logs all the timings (and doesn't happen often)
                print("download")
                send("download", "start", {})
                model = loadModel(model_id, False, precision=model_revision)  # download
                send("download", "done", {})

            print("load")
            model = loadModel(model_id, True, precision=model_revision)  # load
            # dir = "models--" + model_id.replace("/", "--") + "--dda"
            dir = os.path.join(MODELS_DIR, normalized_model_id)
            model.save_pretrained(dir, safe_serialization=True)

            # This is all duped from train_dreambooth, need to refactor TODO XXX
            send("compress", "start", {})
            subprocess.run(
                f"tar cvf - -C {dir} . | zstd -o {model_file}",
                shell=True,
                check=True,  # TODO, rather don't raise and return an error in JSON
            )

            send("compress", "done")
            subprocess.run(["ls", "-l", model_file])

            send("upload", "start", {})
            upload_result = storage.upload_file(model_file, filename)
            send("upload", "done")
            print(upload_result)
            os.remove(model_file)

            # leave model dir for future loads... make configurable?
            # shutil.rmtree(dir)

            # TODO, swap directories, inside HF's cache structure.

        return

    # do a dry run of loading the huggingface model, which will download weights at build time
    # For local dev & preview deploys, download all the models (terrible for serverless deploys)
    if MODEL_ID == "ALL":
        for MODEL_I in MODEL_IDS:
            loadModel(MODEL_I, False, precision=model_revision)
    else:
        loadModel(normalized_model_id, False, precision=model_revision)

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
    download_model("", MODEL_ID, PRECISION)
