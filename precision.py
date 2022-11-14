import os
import torch

PRECISION = os.getenv("PRECISION")

revision = None if PRECISION == "" else PRECISION
torch_dtype = None if PRECISION == "" else torch.float16
