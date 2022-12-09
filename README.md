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
* S3 support, dreambooth training.

Note: This image was created for [kiri.art](https://kiri.art/).
Everything is open source but there may be certain request / response
assumptions.  If anything is unclear, please open an issue.

## Updates and Help

* [Official `docker-diffusers-api` Forum](https://banana-forums.dev/c/open-source/docker-diffusers-api/16):
  help, updates, discussion.
* Subscribe ("watch") these forum topics for:
  * [notable **`main`** branch updates](https://banana-forums.dev/t/official-releases-main-branch/35)
  * [notable **`dev`** branch updates](https://banana-forums.dev/t/development-releases-dev-branch/53)
* Always [check the CHANGELOG](./CHANGELOG.md) for important updates when upgrading.

**Official help in our dedicated forum https://banana-forums.dev/c/open-source/docker-diffusers-api/16.**

*[See the `dev` branch for the latest features.](https://github.com/kiri-art/docker-diffusers-api/tree/dev)
**Pull Requests must be submitted against the dev branch.***

## Usage:

Firstly, fork and clone this repo.

Most of the configuration happens via docker build variables.  You can
see all the options in the [Dockerfile](./Dockerfile), and edit them
there directly, or set via docker command line or e.g. Banana's dashboard
UI once support for build variables land (any day now).

If you're only deploying one container, that's all you need!  If you
intend to deploy multiple containers each with different variables
(e.g. a few different models), you can edit the example
[`scripts/permutations.yaml`](scripts/permutations.yaml)] file and
run [`scripts/permute.sh`](scripts/permute.sh) to create a number
of sub-repos in the `permutations` directory.

Lastly, there's an option to set `MODEL_ID=ALL`, and *all* models will
be downloaded, and switched at request time (great for dev, useless for
serverless).

**Deploying to banana?** That's it!  You're done.  Commit your changes and push.

## Running locally / development:

**Building**

1. `docker build -t banana-sd --build-arg HF_AUTH_TOKEN=$HF_AUTH_TOKEN .`
1. See [CONTRIBUTING.md](./CONTRIBUTING.md) for more helpful hints.
1. Note: your first build can take a really long time, depending on
    your PC & network speed, and *especially when using the `CHECKPOINT_URL`
    feature*.  Great time to grab a coffee or take a walk.

**Running**

1. `docker run -it --gpus all -p 8000:8000 banana-sd`
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
    "MODEL_ID": "runwayml/stable-diffusion-v1-5",
    "PIPELINE": "StableDiffusionPipeline",
    "SCHEDULER": "LMSDiscreteScheduler",
    "safety_checker": true,
  },
}
```

If you're using banana's SDK, it looks something like this:

```js
const out = await banana.run(apiKey, modelKey, { "modelInputs": modelInputs, "callInputs": callInputs });
```

NB: if you're coming from another banana starter repo, note that we
explicitly name `modelInputs` above, and send a bigger object (with
`modelInputs` and `callInputs` keys) for the banana-sdk's
"modelInputs" argument.

If provided, `init_image` and `mask_image` should be base64 encoded.

**Schedulers**: docker-diffusers-api is simply a wrapper around diffusers,
literally any scheduler included in diffusers will work out of the box,
provided it can loaded with its default config and without requiring
any other explicit arguments at init time.  In any event, the following
schedulers are the most common and most well tested:
`DPMSolverMultistepScheduler` (fast!  only needs 20 steps!),
`LMSDiscreteScheduler`, `DDIMScheduler`, `PNDMScheduler`,
`EulerAncestralDiscreteScheduler`, `EulerDiscreteScheduler`.

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



## Troubleshooting

* **403 Client Error: Forbidden for url**

  Make sure you've accepted the license on the model card of the HuggingFace model
  specified in `MODEL_ID`, and that you correctly passed `HF_AUTH_TOKEN` to the
  container.

## Adding other Models

You have two options.

1. For a diffusers model, simply set the `MODEL_ID` docker build variable to the name
  of the model hosted on HuggingFace, and it will be downloaded automatically at
  build time.

1. For a non-diffusers model, simply set the `CHECKPOINT_URL` docker build variable
  to the URL of a `.ckpt` file, which will be downloaded and converted to the diffusers
  format automatically at build time.

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

This info is now logged regardless, and `init()` and `inference()` times are sent
back via `{ $timings: { init: timeInMs, inference: timeInMs } }`.

## Acknowledgements

Originally based on https://github.com/bananaml/serverless-template-stable-diffusion.

