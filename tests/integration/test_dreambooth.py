from .lib import getMinio, getDDA
from test import runTest


class TestDreamBoothS3:
    """
    Train/Infer via S3 model save.
    """

    def setup_class(self):
        print("setup_class")
        self.minio = getMinio("global")

    def teardown_class(self):
        print("teardown_class")
        self.minio.stop()

    def test_training_s3(self):
        dda = getDDA(
            minio=self.minio,
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
                "dest_url": f"s3:///{self.minio.aws_s3_default_bucket}/model.tar.zst",
            },
            {"max_train_steps": 1},
        )

        dda.stop()
        timings = result["$timings"]
        assert timings["training"] > 0
        assert timings["upload"] > 0

    # dependent on above, TODO, mark as such.
    def test_s3_download_and_inference(self):
        dda = getDDA(
            minio=self.minio,
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
                "MODEL_URL": f"s3:///{self.minio.aws_s3_default_bucket}/model.tar.zst",
            },
            {"num_inference_steps": 1},
        )

        dda.stop()
        assert result["image_base64"]
