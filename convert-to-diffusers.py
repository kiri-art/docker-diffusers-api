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
    print("Converting " + fname + " to diffusers model " + MODEL_ID + "...", flush=True)

    subprocess.run(["pip", "install", "omegaconf", "pytorch_lightning"])
    subprocess.run(["apt-get", "install", "-y", "wget"])
    subprocess.run(
        [
            "sed",
            "-i",
            # Force loading into CPU
            "s/torch.load(args.checkpoint_path)/torch.load(args.checkpoint_path, map_location=torch.device('cpu'))/",
            "./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py",
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
            "./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py",
        ]
    )
    subprocess.run(
        [
            "python3",
            "./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py",
            "--extract_ema",
            "--checkpoint_path",
            fname,
            "--dump_path",
            MODEL_ID,
        ]
    )
