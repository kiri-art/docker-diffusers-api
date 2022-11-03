# docker-diffusers-api ("banana-sd-base")

Diffusers / Stable Diffusion in docker with a REST API, supporting various models, pipelines & schedulers.  Used by [kiri.art](https://kiri.art/), perfect for [banana.dev](https://www.banana.dev/).

Copyright (c) Gadi Cohen, 2022.  MIT Licensed.
Please give credit and link back to this repo if you use it in a public project.

## Features

* Pipelines: txt2img, img2img and inpainting in a single container
* Models: stable-diffusion, waifu-diffusion, and easy to add others (e.g. jp-sd)
* All model inputs supported, including setting nsfw filter per request
* *Permute* base config to multiple forks based on yaml config with vars
* Optionally send signed event logs / performance data to a REST endpoint
* Can automatically download a checkpoint file and convert to diffusers.

Note: This image was created for [kiri.art](https://kiri.art/).
Everything is open source but there may be certain request / response
assumptions.  If anything is unclear, please open an issue.

## [Read the CHANGELOG for Important Updates.](./CHANGELOG.md)

## Usage:

1. Clone or fork this repo.

1. **Variables**:

    1. *EITHER*:
        1. Set in `Dockerfile`.
    2. *OR*:
        1. Set `HF_AUTH_TOKEN` environment variable,
        1. Edit `scripts/permutations.yaml`,
        1. Run `scripts/permute.sh` to create a bunch of distinct forks.
    3. *DEV MODE*:
        1. Leave `MODEL_ID` as `ALL` and *all* models will be downloaded,
           (as listed in [`loadModel.py`](./loadModel.py)) allowing you
           to switch at request time (great for dev, useless for serverless).

1. **Building**

    1. Set `HF_AUTH_TOKEN` environment var if you haven't set it elsewhere.
    1. `docker build -t banana-sd --build-arg HF_AUTH_TOKEN=$HF_AUTH_TOKEN .`
    1. Optionally add `DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain docker` to
       start of the line, depending on your preferences.  (Recommended if
       you're using the `root-cache` feature.)

1. **Running**

    1. `docker run -it --gpus all -p 8000:8000 banana-sd python3 server.py`
    1. Note: the `-it` is optional but makes it alot quicker/easier to stop the
       container using `Ctrl-C`.
    1. If you get a `CUDA initialization: CUDA unknown error` after suspend,
       just stop the container, `rmmod nvidia_uvm`, and restart.

## Sending requests

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
    "MODEL_ID": "CompVis/stable-diffusion-v1-4",
    "PIPELINE": "StableDiffusionPipeline",
    "SCHEDULER": "LMS",
    "safety_checker": true,
  },
}
```

If provided, `init_image` and `mask_image` should be base64 encoded.

If you're using banana's SDK, it looks something like this:

```js
const out = await banana.run(apiKey, modelKey, { modelInputs, callInputs });
```

NB: if you're coming from another banana starter repo, note that we
explicitly name `modelInputs` above, and send a bigger object (with
`modelInputs` and `callInputs` keys) for the banana-sdk's
"modelInputs" argument.

There are also very basic examples in [test.py](./test.py), which you can view
and call `python test.py` if the container is already running on port 8000.

The best example of course is https://kiri.art/ and it's
[source code](https://github.com/kiri-art/stable-diffusion-react-nextjs-mui-pwa).

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
