# docker-diffusers-api ("banana-sd-base")

Diffusers / Stable Diffusion in docker with a REST API, supporting various models, pipelines & schedulers.  Used by [kiri.art](https://kiri.art/), perfect for local, server & serverless.

Copyright (c) Gadi Cohen, 2022.  MIT Licensed.
Please give credit and link back to this repo if you use it in a public project.

[![Docker](https://img.shields.io/docker/v/gadicc/diffusers-api)](https://hub.docker.com/r/gadicc/diffusers-api/tags) [![CircleCI](https://circleci.com/gh/kiri-art/docker-diffusers-api/tree/split.svg?style=svg)](https://circleci.com/gh/kiri-art/docker-diffusers-api?branch=split) [![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg)](https://github.com/semantic-release/semantic-release) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

## Features

* Models: stable-diffusion, waifu-diffusion, and easy to add others (e.g. jp-sd)
* Pipelines: txt2img, img2img and inpainting in a single container

  (
  [all diffusers official and community pipelines](https://forums.kiri.art/t/all-your-pipelines-are-belong-to-us/83) are wrapped, but untested)
* All model inputs supported, including setting nsfw filter per request
* *Permute* base config to multiple forks based on yaml config with vars
* Optionally send signed event logs / performance data to a REST endpoint
* Can automatically download a checkpoint file and convert to diffusers.
* S3 support, dreambooth training.

Note: This image was created for [kiri.art](https://kiri.art/).
Everything is open source but there may be certain request / response
assumptions.  If anything is unclear, please open an issue.

## Updates and Help

* [Official `docker-diffusers-api` Forum](https://forums.kiri.art/c/docker-diffusers-api/16):
  help, updates, discussion.
* Subscribe ("watch") these forum topics for:
  * [notable **`main`** branch updates](https://forums.kiri.art/t/official-releases-main-branch/35)
  * [notable **`dev`** branch updates](https://forums.kiri.art/t/development-releases-dev-branch/53)
* Always [check the CHANGELOG](./CHANGELOG.md) for important updates when upgrading.

**Official help in our dedicated forum https://forums.kiri.art/c/docker-diffusers-api/16.**

*[See the `dev` branch for the latest features.](https://github.com/kiri-art/docker-diffusers-api/tree/dev)
**Pull Requests must be submitted against the dev branch.***

## Installation & Setup:

Setup varies depending on your use case.

1. **To run locally or on a *server*, with runtime downloads:**:

  `docker run --gpus all -p 8000:8000 -e HF_AUTH_TOKEN=$HF_AUTH_TOKEN gadicc/diffusers-api`.

  See the [guides for various cloud providers](https://forums.kiri.art/t/running-on-other-cloud-providers/89/7).

1. **To run *serverless*, include the model at build time**.

  1. [docker-diffusers-api-build-download](https://github.com/kiri-art/docker-diffusers-api-build-download) (
    [banana](https://forums.kiri.art/t/run-diffusers-api-on-banana-dev/103), others)
  1. [docker-diffusers-api-runpod](https://github.com/kiri-art/docker-diffusers-api-runpod),
  see the [guide](https://forums.kiri.art/t/run-diffusers-api-on-runpod-io/102)

1. **Building from source**.

  1. Fork / clone this repo.
  1. `docker build -t gadicc/diffusers-api .`
  1. See [CONTRIBUTING.md](./CONTRIBUTING.md) for more helpful hints.

*Other configurations are possible but these are the most common cases*

Everything is set via docker build-args or environment variables.

## Usage:

See also [Testing](#testing) below.

The container expects an `HTTP POST` request with the following JSON body:

```json
{
  "modelInputs": {
    "prompt": "Super dog",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "seed": 3239022079
  },
  "callInputs": {
    // You can leave these out to use the default
    "MODEL_ID": "runwayml/stable-diffusion-v1-5",
    "PIPELINE": "StableDiffusionPipeline",
    "SCHEDULER": "LMSDiscreteScheduler",
    "safety_checker": true,
  },
}
```

**Schedulers**: docker-diffusers-api is simply a wrapper around diffusers,
literally any scheduler included in diffusers will work out of the box,
provided it can loaded with its default config and without requiring
any other explicit arguments at init time.  In any event, the following
schedulers are the most common and most well tested:
`DPMSolverMultistepScheduler` (fast!  only needs 20 steps!),
`LMSDiscreteScheduler`, `DDIMScheduler`, `PNDMScheduler`,
`EulerAncestralDiscreteScheduler`, `EulerDiscreteScheduler`.

**Pipelines**:

<a name="testing"></a>
## Examples and testing

There are also very basic examples in [test.py](./test.py), which you can view
and call `python test.py` if the container is already running on port 8000.
You can also specify a specific test, change some options, and run against a
deployed banana image:

```bash
$ python test.py
Usage: python3 test.py [--banana] [--xmfe=1/0] [--scheduler=SomeScheduler] [all / test1] [test2] [etc]

# Run against http://localhost:8000/ (Nvidia Quadro RTX 5000)
$ python test.py txt2img
Running test: txt2img
Request took 5.9s (init: 3.2s, inference: 5.9s)
Saved /home/dragon/www/banana/banana-sd-base/tests/output/txt2img.png

# Run against deployed banana image (Nvidia A100)
$ export BANANA_API_KEY=XXX
$ BANANA_MODEL_KEY=XXX python3 test.py --banana txt2img
Running test: txt2img
Request took 19.4s (init: 2.5s, inference: 3.5s)
Saved /home/dragon/www/banana/banana-sd-base/tests/output/txt2img.png

# Note that 2nd runs are much faster (ignore init, that isn't run again)
Request took 3.0s (init: 2.4s, inference: 2.1s)
```

The best example of course is https://kiri.art/ and it's
[source code](https://github.com/kiri-art/stable-diffusion-react-nextjs-mui-pwa).

## Help on [Official Forums](https://forums.kiri.art/c/docker-diffusers-api/16).

## Adding other Models

You have two options.

1. For a diffusers model, simply set `MODEL_ID` build-var / call-arg to the name
  of the model hosted on HuggingFace, and it will be downloaded automatically at
  build time.

1. For a non-diffusers model, simply set the `CHECKPOINT_URL` build-var / call-arg
  to the URL of a `.ckpt` file, which will be downloaded and converted to the diffusers
  format automatically at build time.  `CHECKPOINT_CONFIG_URL` can also be set.

## Troubleshooting

* **403 Client Error: Forbidden for url**

  Make sure you've accepted the license on the model card of the HuggingFace model
  specified in `MODEL_ID`, and that you correctly passed `HF_AUTH_TOKEN` to the
  container.

## Event logs / performance data

Set `CALL_URL` and `SIGN_KEY` environment variables to send timing data on `init`
and `inference` start and end data.  You'll need to check the source code of here
and sd-mui as the format is in flux.

This info is now logged regardless, and `init()` and `inference()` times are sent
back via `{ $timings: { init: timeInMs, inference: timeInMs } }`.

## Acknowledgements

Originally based on https://github.com/bananaml/serverless-template-stable-diffusion.

