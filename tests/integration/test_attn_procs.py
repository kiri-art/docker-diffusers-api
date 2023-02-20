import sys
import os
from .lib import getMinio, getDDA
from test import runTest


class TestAttnProcs:
    def test_hf_download(self):
        """
        Make sure when switching models we release VRAM afterwards.
        """
        dda = getDDA(
            stream_logs=True,
        )
        print(dda)

        TEST_ARGS = {"test_url": dda.url}

        mem_usage = list()

        # fp32 model is obviously bigger
        result = runTest(
            "txt2img",
            TEST_ARGS,
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
        dda.stop()
