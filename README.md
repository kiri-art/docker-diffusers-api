# docker-diffusers-api ("banana-sd-base")

Diffusers / Stable Diffusion in docker with a REST API, supporting various models, pipelines & schedulers.  Used by [kiri.art](https://kiri.art/), perfect for local, server & serverless.

[![Docker](https://img.shields.io/docker/v/gadicc/diffusers-api?sort=semver)](https://hub.docker.com/r/gadicc/diffusers-api/tags) [![CircleCI](https://img.shields.io/circleci/build/github/kiri-art/docker-diffusers-api/split)](https://circleci.com/gh/kiri-art/docker-diffusers-api?branch=split) [![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg)](https://github.com/semantic-release/semantic-release) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Copyright (c) Gadi Cohen, 2022.  MIT Licensed.
Please give credit and link back to this repo if you use it in a public project.

## Features

* Models: stable-diffusion, waifu-diffusion, and easy to add others (e.g. jp-sd)
* Pipelines: txt2img, img2img and inpainting in a single container
  ([all diffusers official and community pipelines](https://forums.kiri.art/t/all-your-pipelines-are-belong-to-us/83) are wrapped, but untested)
* All model inputs supported, including setting nsfw filter per request
* *Permute* base config to multiple forks based on yaml config with vars
* Optionally send signed event logs / performance data to a REST endpoint / webhook.
* Can automatically download a checkpoint file and convert to diffusers.
* S3 support, dreambooth training.

Note: This image was created for [kiri.art](https://kiri.art/).
Everything is open source but there may be certain request / response
assumptions.  If anything is unclear, please open an issue.

## Important Notices

* [Official `docker-diffusers-api` Forum](https://forums.kiri.art/c/docker-diffusers-api/16):
  help, updates, discussion.
* Subscribe ("watch") these forum topics for:
  * [notable **`main`** branch updates](https://forums.kiri.art/t/official-releases-main-branch/35)
  * [notable **`dev`** branch updates](https://forums.kiri.art/t/development-releases-dev-branch/53)
* Always [check the CHANGELOG](./CHANGELOG.md) for important updates when upgrading.

**Official help in our dedicated forum https://forums.kiri.art/c/docker-diffusers-api/16.**

**This README refers to the in-development `dev` branch** and may
reference features and fixes not yet in the published releases.

**`v1` has not yet been officially released yet** but has been
running well in production on kiri.art for almost a month.  We'd
be grateful for any feedback from early adopters to help make
this official.  For more details, see [Upgrading from v0 to
v1](https://forums.kiri.art/t/wip-upgrading-from-v0-to-v1/116).
Previous releases available on the `dev-v0-final` and
`main-v0-final` branches.

**Currently only NVIDIA / CUDA devices are supported**.  Tracking
Apple / M1 support in issue
[#20](https://github.com/kiri-art/docker-diffusers-api/issues/20).

## Installation & Setup:

Setup varies depending on your use case.

1. **To run locally or on a *server*, with runtime downloads:**

    `docker run --gpus all -p 8000:8000 -e HF_AUTH_TOKEN=$HF_AUTH_TOKEN gadicc/diffusers-api`.

    See the [guides for various cloud providers](https://forums.kiri.art/t/running-on-other-cloud-providers/89/7).

1. **To run *serverless*, include the model at build time:**

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

The container expects an `HTTP POST` request to `/`, with a JSON body resembling the following:

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

It's important to remember that `docker-diffusers-api` is primarily a wrapper
around HuggingFace's
[diffusers](https://huggingface.co/docs/diffusers/index) library.
**Basic familiarity with `diffusers` is indespensible for a good experience
with `docker-diffusers-api`.**  Explaining some of the options above:

* **modelInputs** - for the most part - are passed directly to the selected
diffusers pipeline unchanged.  So, for the default `StableDiffusionPipeline`,
you can see all options in the relevant pipeline docs for its
[`__call__`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__) method.  The main exceptions are:

    * Only valid JSON values can be given (strings, numbers, etc)
    * **seed**, a number, is transformed into a `generator`.
    * **images** are converted to / from base64 encoded strings.

* **callInputs** affect which model, pipeline, scheduler and other lower
level options are used to construct the final pipeline.  Notably:

    * **`SCHEDULER`**: any scheduler included in diffusers should work out
    the box, provided it can loaded with its default config and without
    requiring any other explicit arguments at init time.  In any event,
    the following schedulers are the most common and most well tested:
    `DPMSolverMultistepScheduler` (fast!  only needs 20 steps!),
    `LMSDiscreteScheduler`, `DDIMScheduler`, `PNDMScheduler`,
    `EulerAncestralDiscreteScheduler`, `EulerDiscreteScheduler`.

    * **`PIPELINE`**: the most common are
    [`StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img),
    [`StableDiffusionImg2ImgPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img),
    [`StableDiffusionInpaintPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint), and the community
    [`lpw_stable_diffusion`](https://forums.kiri.art/t/lpw-stable-diffusion-pipeline-longer-prompts-prompt-weights/82)
    which allows for long prompts (more than 77 tokens) and prompt weights
    (things like `((big eyes))`, `(red hair:1.2)`, etc), and accepts a
    `custom_pipeline_method` callInput with values `text2img` ("text", not "txt"),
    `img2img` and `inpaint`.  See these links for all the possible `modelInputs`'s
    that can be passed to the pipeline's `__call__` method.

    * **`MODEL_URL`** (optional) can be used to retrieve the model from
    locations other than HuggingFace, e.g. an `HTTP` server, S3-compatible
    storage, etc.  For more info, see the
    [storage docs](https://github.com/kiri-art/docker-diffusers-api/blob/dev/docs/storage.md)
    and
    [this post](https://forums.kiri.art/t/safetensors-our-own-optimization-faster-model-init/98)
    for info on how to use and store optimized models from your own cloud.

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

## Event logs / web hooks / performance data

Set `SEND_URL` (and optionally `SIGN_KEY`) environment variable(s) to send
event and timing data on `init`, `inference` and other start and end events.
This can either be used to log performance data, or for webhooks on event
start / finish.

The timing data is now returned in the response payload too, like this:
`{ $timings: { init: timeInMs, inference: timeInMs } }`, with any other
events (such a `training`, `upload`, etc).

You can go to https://webhook.site/ and use the provided "unique URL"
as your `SEND_URL` to see how it works, if you don't have your own
REST endpoint (yet).

If `SIGN_KEY` is used, you can verify the signature like this (TypeScript):

```ts
import crypto from "crypto";

async function handler(req: NextApiRequest, res: NextApiResponse) {
  const data = req.body;

  const containerSig = data.sig as string;
  delete data.sig;

  const ourSig = crypto
    .createHash("md5")
    .update(JSON.stringify(data) + process.env.SIGN_KEY)
    .digest("hex");

  const signatureIsValid = containerSig === ourSig;
}
```

## Acknowledgements

* The container image is originally based on
  https://github.com/bananaml/serverless-template-stable-diffusion.

* [CompVis](https://github.com/CompVis),
  [Stability AI](https://stability.ai/),
  [LAION](https://laion.ai/)
  and [RunwayML](https://runwayml.com/)
  for their incredible time, work and efforts in creating Stable Diffusion,
  and no less so, their decision to release it publicly with an open source
  license.

* [HuggingFace](https://huggingface.co/) - for their passion and inspiration
  for making machine learning more accessibe to developers, and in particular,
  their [Diffusers](https://github.com/huggingface/diffusers) library.
