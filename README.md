# banana-sd-base

Banana image with multiple Stable Diffusion models, pipelines, schedulers in
a single base that can be automatically forked in various permutations.
Used by
[sd-mui.vercel.app](https://sd-mui.vercel.app)
([github](https://github.com/gadicc/stable-diffusion-react-nextjs-mui-pwa)).
*"One repo to rule them all" :)*

Copyright (c) Gadi Cohen, 2022.  MIT Licensed.
Please give credit and link back to this repo if you use it in a public project.

## Features

* Pipelines: txt2img, img2img and inpainting in a single container
* Models: stable-diffusion, waifu-diffusion, and easy to add others (e.g. jp-sd)
* All model inputs supported, including setting nsfw filter per request
* *Permute* base config to multiple forks based on yaml config with vars
* Optionally send signed event logs / performance data to a REST endpoint

Note: This image is created for [sd-mui.vercel.app](https://sd-mui.vercel.app).
Everything is open source but there may be certain request / response
assumptions.

## Usage:

1. Clone or fork this repo.

1. **Variables**:
    1. *EITHER*:
        1. Set in `DOWNLOAD_VARS.py`, `APP_VARS.py` and `Dockerfile`;
    2. *OR*:
        1. Set `HF_AUTH_TOKEN` environment variable,
        1. Edit `scripts/permutations.yaml`,
        1. Run `scripts/permute.sh` to create a bunch of distinct forks.

2. **Dev mode**:
    1. Leave `MODEL_ID` as `ALL` and *all* models will be downloaded,
    allowing you to switch at request time (great for dev, useless for
    serverless).
    1. Set `HF_AUTH_TOKEN` environment var and run
    `docker build -t banana-sd --build-arg HF_AUTH_TOKEN=$HF_AUTH_TOKEN .`
    1. `docker run --gpus all -p 8000:8000 banana-sd`

## Sending requests

See [sd-mui source](https://github.com/gadicc/stable-diffusion-react-nextjs-mui-pwa)
for more info, but basically, it's:

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
    "MODEL_ID": "CompVis/stable-diffusion-v1-4",
    "PIPELINE": "StableDiffusionPipeline",
    "SCHEDULER": "LMS",
    "safety_checker": true,
  },
}
```

If provided, `init_image` and `mask_image` should be base64 encoded.

Sorry, but this format might change without notice based on the needs of SD-MUI.
It's been stable for a while but we make no promises.  Your best bet is always to
keep up-to-date with the SD-MUI source.

## Keeping forks up to date

Per your personal preferences, rebase or merge, e.g.

1. `git fetch upstream`
1. `git merge upstream/main`
1. `git push`

Or, if you're confident, do it in one step with no confirmations:

  `git fetch upstream && git merge upstream/main --no-edit && git push`

Check `scripts/permute.sh` and your git remotes, some URLs are hardcoded, I'll
make this easier in a future release.

## Event logs / performance data

Set `CALL_URL` and `SIGN_KEY` environment variables to send timing data on `init`
and `inference` start and end data.  You'll need to check the source code of here
and sd-mui as the format is in flux.

***Original Template README follows***

# 🍌 Banana Serverless

This repo gives a basic framework for serving Stable Diffusion in production using simple HTTP servers.

## Quickstart:

1. Create your own private repo and copy the files from this template repo into it. You'll want a private repo so that your huggingface keys are secure.

2. Install the [Banana Github App](https://github.com/apps/banana-serverless) to your new repo.

3. Login in to the [Banana Dashboard](https://app.banana.dev) and setup your account by saving your payment details and linking your Github.

4. Create huggingface account to get permission to download and run [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) text-to-image model.
  - Accept terms and conditions for the use of the v1-4 [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4)

5. Edit the `dockerfile` in your forked repo with `ENV HF_AUTH_TOKEN=your_auth_token`

6. Push that repo to main.

From then onward, any pushes to the default repo branch (usually "main" or "master") trigger Banana to build and deploy your server, using the Dockerfile.
Throughout the build we'll sprinkle in some secret sauce to make your server extra snappy 🔥

It'll then be deployed on our Serverless GPU cluster and callable with any of our serverside SDKs:

- [Python](https://github.com/bananaml/banana-python-sdk)
- [Node JS / Typescript](https://github.com/bananaml/banana-node-sdk)
- [Go](https://github.com/bananaml/banana-go)

You can monitor buildtime and runtime logs by clicking the logs button in the model view on the [Banana Dashboard](https://app.banana.dev)

<br>

## Use Banana for scale.
