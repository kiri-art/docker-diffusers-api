#!/bin/sh

# need to fix this.
#download_model {'model_url': 's3://', 'model_id': 'Linaqruf/anything-v3.0', 'model_revision': 'fp16', 'hf_model_id': None}
# {'normalized_model_id': 'models--Linaqruf--anything-v3.0--fp16'}
#self.endpoint_url https://6fb830ebb3c8fed82a52524211d9c54e.r2.cloudflarestorage.com/diffusers
#Downloading s3:// to /root/.cache/diffusers-api/models--Linaqruf--anything-v3.0--fp16.tar.zst...


MODELS=(
  # ID,precision,revision
#  "prompthero/openjourney-v2"
#  "wd-1-4-anime_e1,,,hakurei/waifu-diffusion"
#  "Linaqruf/anything-v3.0,fp16,diffusers"
#  "Linaqruf/anything-v3.0,fp16,fp16"
#  "stabilityai/stable-diffusion-2-1,fp16,fp16"
#  "stabilityai/stable-diffusion-2-1-base,fp16,fp16"
#  "stabilityai/stable-diffusion-2,fp16,fp16"
#  "stabilityai/stable-diffusion-2-base,fp16,fp16"
#  "CompVis/stable-diffusion-v1-4,fp16,fp16"
#  "runwayml/stable-diffusion-v1-5,fp16,fp16"
#  "runwayml/stable-diffusion-inpainting,fp16,fp16"
#  "hakurei/waifu-diffusion,fp16,fp16"
#  "hakurei/waifu-diffusion-v1-3,fp16,fp16" # from checkpoint
#  "rinna/japanese-stable-diffusion"
#  "OrangeMix/AbyssOrangeMix2,fp16"
#  "OrangeMix/ElyOrangeMix,fp16"
#  "OrangeMix/EerieOrangeMix,fp16"
#  "OrangeMix/BloodOrangeMix,fp16"
  "hakurei/wd-1-5-illusion-beta3,fp16,fp16"
  "hakurei/wd-1-5-ink-beta3,fp16,fp16"
  "hakurei/wd-1-5-mofu-beta3,fp16,fp16"
  "hakurei/wd-1-5-radiance-beta3,fp16,fp16",
)

for MODEL_STR in ${MODELS[@]}; do
  IFS="," read -ra DATA <<<$MODEL_STR
  MODEL_ID=${DATA[0]}
  MODEL_PRECISION=${DATA[1]}
  MODEL_REVISION=${DATA[2]}
  HF_MODEL_ID=${DATA[3]}
  python test.py txt2img \
    --call-arg MODEL_ID="$MODEL_ID" \
    --call-arg HF_MODEL_ID="$HF_MODEL_ID" \
    --call-arg MODEL_PRECISION="$MODEL_PRECISION" \
    --call-arg MODEL_REVISION="$MODEL_REVISION" \
    --call-arg MODEL_URL="s3://" \
    --model-arg num_inference_steps=1
done
