import os
import torch

DEPRECATED_PRECISION = os.getenv("PRECISION")
MODEL_PRECISION = os.getenv("MODEL_PRECISION") or DEPRECATED_PRECISION
MODEL_REVISION = os.getenv("MODEL_REVISION")

if DEPRECATED_PRECISION:
    print("Warning: PRECISION variable been deprecated and renamed MODEL_PRECISION")
    print("Your setup still works but in a future release, this will throw an error")

if MODEL_PRECISION and not MODEL_REVISION:
    print("Warning: we no longer default to MODEL_REVISION=MODEL_PRECISION, please")
    print(f'explicitly set MODEL_REVISION="{MODEL_PRECISION}" if that\'s what you')
    print("want.")


def revision_from_precision(precision=MODEL_PRECISION):
    # return precision if precision else None
    raise Exception("revision_from_precision no longer supported")


def torch_dtype_from_precision(precision=MODEL_PRECISION):
    if precision == "fp16":
        return torch.float16
    return None


def torch_dtype_from_precision(precision=MODEL_PRECISION):
    if precision == "fp16":
        return torch.float16
    return None
