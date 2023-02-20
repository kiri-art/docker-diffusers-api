import sys
import os
from .lib import getMinio, getDDA
from test import runTest


class TestAttnProcs:
    def setup_class(self):
        print("setup_class")
        # self.minio = minio = getMinio("global")

        self.dda = dda = getDDA(
            # minio=minio
            stream_logs=True,
        )
        print(dda)

        self.TEST_ARGS = {"test_url": dda.url}

    def teardown_class(self):
        print("teardown_class")
        # self.minio.stop() - leave global up
        self.dda.stop()

    def test_hf_download(self):
        """
        Download user/repo from HuggingFace.
        """
        # fp32 model is obviously bigger
        result = runTest(
            "txt2img",
            self.TEST_ARGS,
            {
                "MODEL_ID": "runwayml/stable-diffusion-v1-5",
                "MODEL_REVISION": "fp16",
                "MODEL_PRECISION": "fp16",
                "attn_procs": "patrickvonplaten/lora_dreambooth_dog_example",
            },
            {
                "num_inference_steps": 1,
                "prompt": "A picture of a sks dog in a bucket",
                "seed": 1,
                "cross_attention_kwargs": {"scale": 0.5},
            },
        )

        assert result["image_base64"]

    def test_http_download_diffusers_archive(self):
        """
        Download user/repo from HuggingFace.
        """

        # fp32 model is obviously bigger
        result = runTest(
            "txt2img",
            self.TEST_ARGS,
            {
                "MODEL_ID": "runwayml/stable-diffusion-v1-5",
                "MODEL_REVISION": "fp16",
                "MODEL_PRECISION": "fp16",
                "attn_procs": "https://huggingface.co/patrickvonplaten/lora_dreambooth_dog_example/resolve/main/pytorch_lora_weights.bin",
            },
            {
                "num_inference_steps": 1,
                "prompt": "A picture of a sks dog in a bucket",
                "seed": 1,
                "cross_attention_kwargs": {"scale": 0.5},
            },
        )

        assert result["image_base64"]
