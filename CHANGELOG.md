# CHANGELOG

* **2022-10-21**

  * Upgrade to **Diffusers 0.6.0**
  
  * Biggest news of the above release is support for
    ["runwayml/stable-diffusion-inpainting"](https://huggingface.co/runwayml/stable-diffusion-inpainting),
    the first official stable diffusion checkpoint fine-tuned on **inpainting**.

    You need to use the above link to *accept the license terms* on the
    model card before you'll be able to download it.  Also, unfortunately,
    this means we're back to having a separate container for inpainting,
    as it's a different model.

  * **Flash attention** now disabled by default.  1) Because it's built on
    an older version of diffusers, but also because 2) I didn't succeed in
    getting much improvement out of it.  Maybe someone else will have better
    luck.  I think you need big batch sizes to really see the benefit, which
    doesn't suit my use case.  But please anyone who figures anything out,
    let us know.