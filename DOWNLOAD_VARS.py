import os

# We set this in Dockerfile now, in anticipation for build env vars
# "CompVis/stable-diffusion-v1-4", "hakurei/waifu-diffusion"
# "ALL" - build all models (great for dev, bad for serverless)
# MODEL_ID = "CompVis/stable-diffusion-v1-4"

MODEL_ID = os.environ.get("MODEL_ID")
