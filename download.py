# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import os
from loadModel import loadModel, MODEL_IDS

MODEL_ID = os.environ.get("MODEL_ID")


def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    # For local dev & preview deploys, download all the models (terrible for serverless deploys)
    if MODEL_ID == "ALL":
        for MODEL_I in MODEL_IDS:
            loadModel(MODEL_I, False)
    else:
        loadModel(MODEL_ID, False)


if __name__ == "__main__":
    download_model()
