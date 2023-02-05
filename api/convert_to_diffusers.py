import os
import requests
import subprocess
import torch
from utils import Storage
from device import device_id

MODEL_ID = os.environ.get("MODEL_ID", None)
CHECKPOINT_DIR = "/root/.cache/checkpoints"
CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL", None)
CHECKPOINT_CONFIG_URL = os.environ.get("CHECKPOINT_CONFIG_URL", None)
# _CONVERT_SPECIAL = os.environ.get("_CONVERT_SPECIAL", None)


def main(model_id: str, checkpoint_url: str, checkpoint_config_url: str):
    fname = CHECKPOINT_DIR + "/" + checkpoint_url.split("/").pop()

    if checkpoint_config_url and checkpoint_config_url != "":
        storage = Storage(checkpoint_config_url)
        configPath = (
            CHECKPOINT_DIR + "/" + checkpoint_url.split("/").pop() + "_config.yaml"
        )
        print(f"Downloading {checkpoint_config_url} to {configPath}...")
        storage.download_file(configPath)

    # specialSrc = "https://raw.githubusercontent.com/hafriedlander/diffusers/stable_diffusion_2/scripts/convert_original_stable_diffusion_to_diffusers.py"
    # specialPath = CHECKPOINT_DIR + "/" + "convert_special.py"
    # if _CONVERT_SPECIAL:
    #     storage = Storage(specialSrc)
    #     print(f"Downloading {specialSrc} to {specialPath}")
    #     storage.download_file(specialPath)

    scriptPath = (
        # specialPath
        # if _CONVERT_SPECIAL
        # else
        "./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py"
    )

    print("Converting " + fname + " to diffusers model " + model_id + "...", flush=True)

    subprocess.run(
        ["pip", "install", "omegaconf", "pytorch_lightning", "tensorboard"], check=True
    )
    subprocess.run(["apt-get", "install", "-y", "wget"], check=True)
    if device_id == "cpu":
        subprocess.run(
            [
                "sed",
                "-i",
                # Force loading into CPU
                "s/torch.load(args.checkpoint_path)/torch.load(args.checkpoint_path, map_location=torch.device('cpu'))/",
                scriptPath,
            ]
        )
    # Nice to check but also there seems to be a race condition here which
    # needs further investigation.  Python docs are clear that subprocess.run()
    # will "Wait for command to complete, then return a CompletedProcess instance."
    # But it really seems as though without the grep in the middle, the script is
    # run before sed completes, or maybe there's some FS level caching gotchas.
    subprocess.run(
        [
            "grep",
            "torch.load",
            scriptPath,
        ],
        check=True,
    )

    args = [
        "python3",
        scriptPath,
        "--extract_ema",
        "--checkpoint_path",
        fname,
        "--dump_path",
        model_id,
    ]

    if checkpoint_config_url:
        args.append("--original_config_file")
        args.append(configPath)

    subprocess.run(
        args,
        check=True,
    )


if __name__ == "__main__":
    # response = requests.get(
    #    "https://github.com/huggingface/diffusers/raw/main/scripts/convert_original_stable_diffusion_to_diffusers.py"
    # )
    # open("convert_original_stable_diffusion_to_diffusers.py", "wb").write(
    #    response.content
    # )

    if CHECKPOINT_URL and CHECKPOINT_URL != "":
        main(MODEL_ID, CHECKPOINT_URL, CHECKPOINT_CONFIG_URL)
