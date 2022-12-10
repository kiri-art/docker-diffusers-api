internal document to gather my thoughts

RUNTIME_DOWNLOADS=1 (must be build arg)
IMAGE_CLOUD_CACHE="s3://" (can be env arg)
CREATE_MISSING=1

e.g. stabilityai/stable-diffusion-2-1-base

1. Try download from IMAGE_CLOUD_CACHE
  1. If found, use.
  2. If not found:
    1. Download from HuggingFace
    2. In a subprocess:
      1. Save with safetesors to tmp directory
      2. Upload to IMAGE_CLOUD_CACHE
      3. Delete original model dir, mv tmp to model dir (for next load)
    1. Run inference with HF model.

FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1-base/refs/main'
