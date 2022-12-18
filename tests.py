from test import runTest


def main():
    runTest(
        "txt2img",
        False,
        {"MODEL_ID": "CompVis/stable-diffusion-v1-4", "MODEL_URL": "s3://"},
        {},
    )


if __name__ == "__main__":
    main()
