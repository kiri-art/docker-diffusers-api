# CHANGELOG

* **NEXT MAIN**

* **2022-11-29**

  * **Diffusers v0.9.0, Stable Diffusion v2.0**.  Models:
      * `"stabilityai/stable-diffusion-2"` - trained on 768x768
      * `"stabilityai/stable-diffusion-2-base"` - trained on 512x512
      * `"stabilityai/stable-diffusion-2-inpainting"` - untested
      * `""stabilityai/stable-diffusion-x4-upscaler"` - untested

    > https://github.com/huggingface/diffusers/releases

    **NB**: SDv2 does not include a safety_checker.  The model itself is
    "safe" (it's much harder to create NSFW content).  Trying to "turn off"
    the (non-existent) safety checker will throw an error, we'll handle this
    more gracefully in a future release.  This also means you can safely
    ignore this warning on loading:

    ```
    You have disabled the safety checker for
    <class diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'>
    by passing safety_checker=None. Ensure that...
    ```

  * **DPMSolverMultistepScheduler**.  Docker-diffusers-api is simply a wrapper
    around diffusers.  We support all the included schedulers out of the box,
    as long as they can init themselves with default arguments.  So, the above
    scheduler was already working, but we didn't mention it before.  I'll just
    quote diffusers:

    > DPMSolverMultistepScheduler is the firecracker diffusers implementation
    of DPM-Solver++, a state-of-the-art scheduler that was contributed by one
    of the authors of the paper. This scheduler is able to achieve great
    quality in as few as 20 steps. It's a drop-in replacement for the default
    Stable Diffusion scheduler, so you can use it to essentially half
    generation times.

  * **Storage Class / S3 support**.  We now have a generic storage class, which
    allows for special URLs anywhere anywhere you can usually specify a URL,
    e.g. `CHECKPOINT_URL`, `dest_url` (after dreambooth training), and the new
    `MODEL_URL` (see below).  URLs like "s3:///bucket/filename" will work how
    you expect, but definitely read [docs/storage.md](./docs/storage.md)
    to understand the format better.  Note in particular the triple forwardslash
    ("///") in the beginning to use the default S3 endpoint.

  * **Dreambooth training**, working but still in development.  See
    [this forum post](https://banana-forums.dev/t/dreambooth-training-first-look/36)
    for more info.

  * **`PRECISION`** build var, defaults to `"fp16"`, set to `""` to use the model
    defaults (generally fp32).

  * **`CHECKPOINT_URL` conversion**:
    * Crash / stop build if conversion fails (rather than unclear errors later on)
    * Force `cpu` loading even for models that would otherwise default to GPU.
      This fixes certain models that previously crashed in build stage (where GPU
      is not available).
    * `--extract-ema` on conversion since these are the more important weights for
      inference.
    * `CHECKPOINT_CONFIG_URL` now let's to specify a specific config file for 
      conversion, to use instead of SD's default `v1-inference.yaml`.

  * **`MODEL_URL`**.  If your model is already in diffusers format, but you don't
    host it on HuggingFace, you can now have it downloaded at build time.  At
    this stage, it should be a `.tar.zst` file.  This is an *alternative* to
    `CHECKPOINT_URL` which downloads a `.ckpt` file and converts to diffusers.

  * **`test.py`**:
    * New `--banana` arg to run the test on banana.  Set environment variables
      `BANANA_API_KEY` and `BANANA_MODEL_KEY` first.
    * You can now add to and override a test's default json payload with:
      * `--model-arg prompt="hello"`
      * `--call-arg MODEL_ID="my-model"`
    * Support for extra timing data (e.g. dreambooth sends `train`
      and `upload` timings).
    * Quit after inference errors, don't keep looping.

  * **Dev: better caching solution**.  No more unruly `root-cache` directory.  See
    [CONTRIBUTING.md](./CONTRIBUTING.md) for more info.

* **2022-11-08**

  * **Much faster `init()` times!**  For `runwayml/stable-diffusion-v1-5`:

    * Previously: 4.0s, now: 2.4s (40% speed gain)

  * **Much faster `inference()` times!** Particularly from the 2nd inference onwards.
    Here's a brief comparison of *inference* average times (for 512x512 x50 steps):

    * [Cold] Previously: 3.8s, now: 3.3s (13% speed gain)
    * [Warm] Previously: 3.2s, now: 2.1s (34% speed gain)

  * **Improved `test.py`**, see [Testing](./README.md#testing)

* **2022-11-05**

  * Upgrade to **Diffusers v0.7.0**.  There is a lot of fun stuff in this release,
    but notably for docker-diffusers-api TODAY (more fun stuff coming next week!),
    we have **much faster init times** (via
    [`fast_load`](https://github.com/huggingface/diffusers/commit/7482178162b779506a54538f2cf2565c8b88c597)
    ) and the greatly anticipated support for the Euler schedulers (
    [a1ea8c0](https://github.com/huggingface/diffusers/commit/a1ea8c01c31a44bf48f6a3b85ccabeb45ef6418f)
    ).

  * We now use the **full scheduler name** for `callInputs.SCHEDULER`.  `"LMS"`,
    `"DDIM"`, `"PNDM"` all still work fine for now but give a deprecation warning
    and will stop working in a future update.  The full list of supported schedulers
    is: `LMSDiscreteScheduler`, `DDIMScheduler`, `PNDMScheduler`,
    `EulerAncestralDiscreteScheduler`, `EulerDiscreteScheduler`.  These cover the
    most commonly used / requested schedulers, but we already have code in place to
    support every scheduler provided by diffusers, which will work in a later
    diffusers release when they have better defaults.

* **2022-10-24**

  * **Fixed img2img and inpainting pipelines**.  To my great shame, in my rush to get
    the new models out before the weekend, I inadvertently broke the above two models.
    Please accept my sincere apology for any confusion this may have caused and
    especially any of your wasted time in debugging this 🙇

  * **Event logs now shown without `SEND_URL`**.  We optionally log useful info at the
    start and end of `init()` and `inference()`.  Previously this was only logged if
    `SEND_URL` was set, to send to an external REST API for logging.  But now, even if
    we don't send it anywhere, we'll still log this useful info.  It now also logs
    the `diffusers` version too.

* **2022-10-21**

  * **Stable Diffusion 1.5 released!!!**

    Accept the license at:
    ["runwayml/stable-diffusion-v1-5"](https://huggingface.co/runwayml/stable-diffusion-v1-5)

    It's the new default model.

  * **Official Stable Diffusion inpainting model**

    Accept the license at:
    ["runwayml/stable-diffusion-inpainting"](https://huggingface.co/runwayml/stable-diffusion-inpainting),

    A few big caveats!

    1) Different model - so back to a separate container for inpainting, also because:
    2) New pipeline that can't share model struct with other pipelines
       (see [diffusers#920](https://github.com/huggingface/diffusers/issues/920)).
    3) Old pipeline is now called `StableDiffusionInpaintPipelineLegacy` (for sd-1.4)
    4) `model_input` takes `image` now, and not `init_image` like the legacy model.
    5) There is no `strength` parameter in the new model
       (see [diffusers#920](https://github.com/huggingface/diffusers/issues/920)).

  * Upgrade to **Diffusers v0.7.0.dev0**

  * **Flash attention** now disabled by default.  1) Because it's built on
    an older version of diffusers, but also because 2) I didn't succeed in
    getting much improvement out of it.  Maybe someone else will have better
    luck.  I think you need big batch sizes to really see the benefit, which
    doesn't suit my use case.  But please anyone who figures anything out,
    let us know.
