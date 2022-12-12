import os
import torch

PRECISION = os.getenv("PRECISION")

revision = None if PRECISION == "" else PRECISION
torch_dtype = None if PRECISION == "" else torch.float16


def revision_from_precision(precision=PRECISION):
    return precision if precision else None


def torch_dtype_from_precision(precision=PRECISION):
    if precision == "fp16":
        return torch.float16
    return None
