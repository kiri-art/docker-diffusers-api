# banana-sd-base

Multiple Stable Diffusion models, pipelines, schedulers in a repo.  Use them all in a single dev image, or set the variables to deploy with a specific permutation.

*"One repo to rule them all" :)*

## Usage:

TODO, see sd-mui source.
But note `HF_AUTH_TOKEN` in Dockerfile, `DOWNLOAD_VARS.py` and `APP_VARS.py`.
`docker build -t banana-sd --build-arg HF_AUTH_TOKEN=$HF_AUTH_TOKEN .`
`git fetch upstream && git merge upstream/main --no-edit && git push`

*Original Template README follows*

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
