import os
import requests
import subprocess

CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL", None)
MODEL_ID = os.environ.get("MODEL_ID", None)

if __name__ == "__main__":
    # response = requests.get(
    #    "https://github.com/huggingface/diffusers/raw/main/scripts/convert_original_stable_diffusion_to_diffusers.py"
    # )
    # open("convert_original_stable_diffusion_to_diffusers.py", "wb").write(
    #    response.content
    # )

    if not CHECKPOINT_URL or CHECKPOINT_URL == "":
        quit()

    CHECKPOINT_DIR = "/root/.cache/checkpoints"
    fname = CHECKPOINT_DIR + "/" + CHECKPOINT_URL.split("/").pop()
    print("Converting " + fname + " to diffusers model " + MODEL_ID + "...")

    subprocess.run(["pip", "install", "omegaconf"])
    subprocess.run(["apt-get", "install", "-y", "wget"])
    subprocess.run(
        [
            "python3",
            "./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py",
            "--checkpoint_path",
            fname,
            "--dump_path",
            MODEL_ID,
        ]
    )
