upsamplers = {
    "RealESRGAN_x4plus": {
        "name": "General - RealESRGANplus",
        "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "net": "RRDBNet",
        "initArgs": {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 23,
            "num_grow_ch": 32,
            "scale": 4,
        },
        "netscale": 4,
    },
    # "RealESRNet_x4plus": {
    #     "name": "",
    #     "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    #     "path": "weights/RealESRNet_x4plus.pth",
    # },
    "RealESRGAN_x4plus_anime_6B": {
        "name": "Anime - anime6B",
        "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "net": "RRDBNet",
        "initArgs": {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 6,
            "num_grow_ch": 32,
            "scale": 4,
        },
        "netscale": 4,
    },
    # "RealESRGAN_x2plus": {
    #     "name": "",
    #     "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    #     "path": "weights/RealESRGAN_x2plus.pth",
    # },
    # "realesr-animevideov3": {
    #     "name": "AnimeVideo - v3",
    #     "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    #    "path": "weights/realesr-animevideov3.pth",
    # },
    "realesr-general-x4v3": {
        "name": "General - v3",
        # [, "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth" ],
        "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "filename": "realesr-general-x4v3.pth",
        "net": "SRVGGNetCompact",
        "initArgs": {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_conv": 32,
            "upscale": 4,
            "act_type": "prelu",
        },
        "netscale": 4,
    },
}

face_enhancers = {
    "GFPGAN": {
        "name": "GFPGAN",
        "weights": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "filename": "GFPGANv1.4.pth",
    },
}

models_by_type = {
    "upsamplers": upsamplers,
    "face_enhancers": face_enhancers,
}
