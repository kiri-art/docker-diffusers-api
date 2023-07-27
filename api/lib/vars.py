import os

RUNTIME_DOWNLOADS = os.getenv("RUNTIME_DOWNLOADS") == "1"
USE_DREAMBOOTH = os.getenv("USE_DREAMBOOTH") == "1"
MODEL_ID = os.environ.get("MODEL_ID")
PIPELINE = os.environ.get("PIPELINE")
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
HOME = os.path.expanduser("~")
MODELS_DIR = os.path.join(HOME, ".cache", "diffusers-api")
