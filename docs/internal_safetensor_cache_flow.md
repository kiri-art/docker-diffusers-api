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


NVIDIA RTX Quadro 5000

NO SAFETENSORS
Downloaded in 462557 ms
Loading model: stabilityai/stable-diffusion-2-1 (fp32)
Loaded from disk in 3113 ms, to gpu in 1644 ms

SAFETENSORS_FAST_GPU=0
Loaded from disk in 2741 ms, to gpu in 557 ms

SAFETENSORS_FAST_GPU=1
Loaded from disk in 1153 ms, to gpu in 1495 ms



NVIDIA RTX Quadro 5000 (fp16)

NO SAFETENSORS
Downloaded in 462557 ms
Loading model: stabilityai/stable-diffusion-2-1-base (fp16)
Loaded from disk in 2043 ms, to gpu in 1539 ms

SAFETENSORS_FAST_GPU=0


SAFETENSORS_FAST_GPU=1
Loaded from disk in 1134 ms, to gpu in 1184 ms
