# CHANGELOG

* **2022-10-21**

  * **Stable Diffusion 1.5 released!!!**

    Accept the license at:
    ["runwayml/stable-diffusion-v1-5"](https://huggingface.co/runwayml/stable-diffusion-v1-5)

    It's the new default model.

  * **Official Stable Diffusion inpainting model**

    Accept the license at:
    ["runwayml/stable-diffusion-inpainting"](https://huggingface.co/runwayml/stable-diffusion-inpainting),

    Unfortunately, as it's a different model, it means we're back to having a
    separate container for inpainting.

    NB: `model_input` takes `image` now, and not `init_image` like the legacy model.

  * Upgrade to **Diffusers v0.7.0.dev0**

  * **Flash attention** now disabled by default.  1) Because it's built on
    an older version of diffusers, but also because 2) I didn't succeed in
    getting much improvement out of it.  Maybe someone else will have better
    luck.  I think you need big batch sizes to really see the benefit, which
    doesn't suit my use case.  But please anyone who figures anything out,
    let us know.