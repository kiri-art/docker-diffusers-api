import os
import asyncio
from pathlib import Path

import base64
from io import BytesIO
import PIL
import json
import cv2
import numpy as np
import torch
import torchvision

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer

from .models import models_by_type, upsamplers, face_enhancers
from status import status
from utils import Storage
from send import send

print(
    {
        "torch.__version__": torch.__version__,
        "torchvision.__version__": torchvision.__version__,
    }
)

HOME = os.path.expanduser("~")
CACHE_DIR = os.path.join(HOME, ".cache", "diffusers-api", "upsample")


def cache_path(filename):
    return os.path.join(CACHE_DIR, filename)


async def assert_model_exists(src, filename, send_opts, opts={}):
    dest = cache_path(filename) if not opts.get("absolutePath", None) else filename
    if not os.path.exists(dest):
        await send("download", "start", {}, send_opts)
        storage = Storage(src, status=status)
        # await storage.download_file(dest)
        await asyncio.to_thread(storage.download_file, dest)
        await send("download", "done", {}, send_opts)


async def download_models(send_opts={}):
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    for type in models_by_type:
        models = models_by_type[type]
        for model_key in models:
            model = models[model_key]
            await assert_model_exists(model["weights"], model["filename"], send_opts)

    Path("gfpgan/weights").mkdir(parents=True, exist_ok=True)

    await assert_model_exists(
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "detection_Resnet50_Final.pth",
        send_opts,
    )
    await assert_model_exists(
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "parsing_parsenet.pth",
        send_opts,
    )

    # hardcoded paths in xinntao/facexlib
    filenames = ["detection_Resnet50_Final.pth", "parsing_parsenet.pth"]
    for file in filenames:
        if not os.path.exists(f"gfpgan/weights/{file}"):
            os.symlink(cache_path(file), f"gfpgan/weights/{file}")


nets = {
    "RRDBNet": RRDBNet,
    "SRVGGNetCompact": SRVGGNetCompact,
}

models = {}


async def upsample(model_inputs, call_inputs, send_opts={}, startRequestId=None):
    global models

    # TODO, only download relevant models for this request
    await download_models()

    model_id = call_inputs.get("MODEL_ID", None)

    if not model_id:
        return {
            "$error": {
                "code": "MISSING_MODEL_ID",
                "message": "call_inputs.MODEL_ID is required, but not given.",
            }
        }

    model = models.get(model_id, None)
    if not model:
        model = models_by_type["upsamplers"].get(model_id, None)
        if not model:
            return {
                "$error": {
                    "code": "MISSING_MODEL",
                    "message": f'Model "{model_id}" not available on this container.',
                    "requested": model_id,
                    "available": '"' + '", "'.join(models.keys()) + '"',
                }
            }
        else:
            modelModel = nets[model["net"]](**model["initArgs"])
            await send(
                "loadModel",
                "start",
                {"startRequestId": startRequestId},
                send_opts,
            )
            upsampler = RealESRGANer(
                scale=model["netscale"],
                model_path=cache_path(model["filename"]),
                dni_weight=None,
                model=modelModel,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )
            await send(
                "loadModel",
                "done",
                {"startRequestId": startRequestId},
                send_opts,
            )
            model.update({"model": modelModel, "upsampler": upsampler})
            models.update({model_id: model})

    upsampler = model["upsampler"]

    input_image = model_inputs.get("input_image", None)
    if not input_image:
        return {
            "$error": {
                "code": "NO_INPUT_IMAGE",
                "message": "Missing required parameter `input_image`",
            }
        }

    if model_id == "realesr-general-x4v3":
        denoise_strength = model_inputs.get("denoise_strength", 1)
        if denoise_strength != 1:
            # wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            # model_path = [model_path, wdn_model_path]
            # upsampler = models["realesr-general-x4v3-denoise"]
            # upsampler.dni_weight = dni_weight
            dni_weight = [denoise_strength, 1 - denoise_strength]
            return "TODO: denoise_strength"

    face_enhance = model_inputs.get("face_enhance", False)
    if face_enhance:
        face_enhancer = models.get("GFPGAN", None)
        if not face_enhancer:
            await send(
                "loadModel",
                "start",
                {"startRequestId": startRequestId},
                send_opts,
            )
            print("1) " + cache_path(face_enhancers["GFPGAN"]["filename"]))
            face_enhancer = GFPGANer(
                model_path=cache_path(face_enhancers["GFPGAN"]["filename"]),
                upscale=4,  # args.outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )
            await send(
                "loadModel",
                "done",
                {"startRequestId": startRequestId},
                send_opts,
            )
            models.update({"GFPGAN": face_enhancer})

    if face_enhance:  # Use GFPGAN for face enhancement
        face_enhancer.bg_upsampler = upsampler

    # image = decodeBase64Image(model_inputs.get("input_image"))
    image_str = base64.b64decode(model_inputs["input_image"])
    image_np = np.frombuffer(image_str, dtype=np.uint8)
    # bytes = BytesIO(base64.decodebytes(bytes(model_inputs["input_image"], "utf-8")))
    img = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)

    await send("inference", "start", {"startRequestId": startRequestId}, send_opts)

    # Run the model
    # with autocast("cuda"):
    #    image = pipeline(**model_inputs).images[0]
    if face_enhance:
        _, _, output = face_enhancer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
    else:
        output, _rgb = upsampler.enhance(img, outscale=4)  # TODO outscale param

    image_base64 = base64.b64encode(cv2.imencode(".jpg", output)[1]).decode()

    await send("inference", "done", {"startRequestId": startRequestId}, send_opts)

    # Return the results as a dictionary
    return {"$meta": {}, "image_base64": image_base64}
