# CHANGELOG

* **2022-11-08**

  * **Much faster `init()` times!**  For `runwayml/stable-diffusion-v1-5`:

    * Previously: 4.0s, now: 2.4s (40% speed gain)

  * **Much faster `inference()` times!** Particularly from the 2nd inference onwards.
    Here's a brief comparison of *inference* average times (for 512x512 x50 steps):

    * [Cold] Previously: 3.8s, now: 3.3s (13% speed gain)
    * [Warm] Previously: 3.2s, now: 2.1s (34% speed gain)

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
