from .lib import getMinio, getDDA, AWS_S3_DEFAULT_BUCKET
from test import runTest


def test_training_s3():
    minio = getMinio("global")
    dda = getDDA(
        minio=minio,
        stream_logs=True,
    )
    print(dda)

    # fp32 model is obviously bigger
    result = runTest(
        "dreambooth",
        {"test_url": dda.url},
        {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
            "MODEL_REVISION": "",
            "MODEL_PRECISION": "",
            "MODEL_URL": "s3://",
            "train": "dreambooth",
            "dest_url": f"s3:///{AWS_S3_DEFAULT_BUCKET}/model.tar.zst",
        },
        {"max_train_steps": 1},
    )

    dda.stop()
    minio.stop()
    timings = result["$timings"]
    assert timings["training"] > 0
    assert timings["upload"] > 0


def test_inference():
    dda = getDDA(
        stream_logs=True,
        root_cache=False,
    )
    print(dda)

    # fp32 model is obviously bigger
    result = runTest(
        "txt2img",
        {"test_url": dda.url},
        {
            "MODEL_ID": "model",
            "MODEL_PRECISION": "fp16",
            "MODEL_URL": f"s3:///{AWS_S3_DEFAULT_BUCKET}/model.tar.zst",
        },
        {"num_inference_steps": 1},
    )

    dda.stop()
    assert result["image_base64"]
