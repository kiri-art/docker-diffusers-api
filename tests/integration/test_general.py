import sys
import os
from .lib import getMinio, getDDA
from test import runTest


class TestGeneralClass:
    """
    Typical usage tests, that assume model is already available locally.
    txt2img, img2img, inpaint.
    """

    CALL_ARGS = {
        "MODEL_ID": "stabilityai/stable-diffusion-2-1-base",
        "MODEL_PRECISION": "fp16",
        "MODEL_REVISION": "fp16",
        "MODEL_URL": "s3://",
    }

    MODEL_ARGS = {"num_inference_steps": 2}

    def setup_class(self):
        print("setup_class")
        self.minio = minio = getMinio("global")

        self.dda = dda = getDDA(
            minio=minio
            # stream_logs=True,
        )
        print(dda)

        self.TEST_ARGS = {"test_url": dda.url}

    def teardown_class(self):
        print("teardown_class")
        # self.minio.stop() - leave global up
        self.dda.stop()

    def test_txt2img(self):
        result = runTest("txt2img", self.TEST_ARGS, self.CALL_ARGS, self.MODEL_ARGS)
        assert result["image_base64"]

    def test_img2img(self):
        result = runTest("img2img", self.TEST_ARGS, self.CALL_ARGS, self.MODEL_ARGS)
        assert result["image_base64"]

    # def test_inpaint(self):
    #     """
    #     This is actually calling inpaint with SDv2.1, not the inpainting model,
    #     so I guess we're testing inpaint-legacy.
    #     """
    #     result = runTest("inpaint", self.TEST_ARGS, self.CALL_ARGS, self.MODEL_ARGS)
    #     assert result["image_base64"]
