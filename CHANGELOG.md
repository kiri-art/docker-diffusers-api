# [1.3.0](https://github.com/kiri-art/docker-diffusers-api/compare/v1.2.2...v1.3.0) (2023-01-26)


### Bug Fixes

* **diffusers:** bump to v0.12.0 ([635d9d9](https://github.com/kiri-art/docker-diffusers-api/commit/635d9d97a010c49ef7875fcb4b43b668848ced0b))
* **diffusers:** update to latest commit ([87632aa](https://github.com/kiri-art/docker-diffusers-api/commit/87632aa2c32faddfeb049fe969884b568066edd3))
* **dreambooth:** bump diffusers, fixes fp16 mixed precision training ([0f5d5ff](https://github.com/kiri-art/docker-diffusers-api/commit/0f5d5ff2bf5b73260b9d60521389f0938f205219))
* **dreambooth:** merge commits to v0.12.0 (NB: mixed-precision issue) ([88f04f8](https://github.com/kiri-art/docker-diffusers-api/commit/88f04f870814aa9baf2a7c09513dcc796070b814))
* **pipelines:** fix clearPipelines() backport from cloud-cache ([9577f93](https://github.com/kiri-art/docker-diffusers-api/commit/9577f9344f0060edc185e32eadeb57e83551aa7f))
* **requirements:** bump transformers,accelerate,safetensors & others ([aebcf65](https://github.com/kiri-art/docker-diffusers-api/commit/aebcf6562808a817e6ee29e88f178f22f54c861b))
* **re:** use raw strings r"" for regexps ([41310c2](https://github.com/kiri-art/docker-diffusers-api/commit/41310c26bbc19069db492781313b162f0fc4d7d9))
* **tests/lambda:** export HF_AUTH_TOKEN ([9f11e7b](https://github.com/kiri-art/docker-diffusers-api/commit/9f11e7b2f0d2a377a44b22d446274677bd025813))
* **test:** shallow copy to avoid mutating base test inputs ([8c41167](https://github.com/kiri-art/docker-diffusers-api/commit/8c41167461308b14066be1472fd8957dc6cdd658))


### Features

* **downloads:** RUNTIME_DOWNLOAD from HF when no MODEL_URL given ([73784a1](https://github.com/kiri-art/docker-diffusers-api/commit/73784a1844ef2b14c628eb399bec0e52661df35c))

## [1.2.2](https://github.com/kiri-art/docker-diffusers-api/compare/v1.2.1...v1.2.2) (2023-01-09)


### Bug Fixes

* **dreambooth:** runtime_dls path fix; integration tests ([ce3827f](https://github.com/kiri-art/docker-diffusers-api/commit/ce3827f6aabd5158c39c99ffae0358d832de2e39))
* **loadModel:** revision = None if revision == "" else revision ([1773631](https://github.com/kiri-art/docker-diffusers-api/commit/1773631e292e28fae20b0a6c93406378aed85d47))

## [1.2.1](https://github.com/kiri-art/docker-diffusers-api/compare/v1.2.0...v1.2.1) (2023-01-05)


### Bug Fixes

* **build-download:** support regular HF download not just cloud cache ([52edf6b](https://github.com/kiri-art/docker-diffusers-api/commit/52edf6b8e52cba4a03c8ea0f72b8fd1e69fa87ad))

# [1.2.0](https://github.com/kiri-art/docker-diffusers-api/compare/v1.1.0...v1.2.0) (2023-01-04)


### Features

* **build:** separate MODEL_REVISION, MODEL_PRECISION, HF_MODEL_ID ([fa9dd16](https://github.com/kiri-art/docker-diffusers-api/commit/fa9dd16b7369d37f3997ef46581df471bca8e7c1))

# [1.1.0](https://github.com/kiri-art/docker-diffusers-api/compare/v1.0.2...v1.1.0) (2023-01-04)


### Features

* **downloads:** allow HF_MODEL_ID call-arg (defauls to MODEL_ID) ([adaa7f6](https://github.com/kiri-art/docker-diffusers-api/commit/adaa7f67aba49058b2e52117e6eb0fed6417b773))
* **downloads:** allow separate MODEL_REVISION and MODEL_PRECISION ([6edc821](https://github.com/kiri-art/docker-diffusers-api/commit/6edc821da1593f34e4502352dba8f2f4cd808e95))

## [1.0.2](https://github.com/kiri-art/docker-diffusers-api/compare/v1.0.1...v1.0.2) (2023-01-01)


### Bug Fixes

* **diffusers:** bump to 2022-12-30 commit 62608a9 ([2f29165](https://github.com/kiri-art/docker-diffusers-api/commit/2f291655967a253b81da9f44c99d4ac68e1c8353))

## [1.0.1](https://github.com/kiri-art/docker-diffusers-api/compare/v1.0.0...v1.0.1) (2022-12-31)


### Bug Fixes

* **ci:** different token, https auth ([ecd0b5d](https://github.com/kiri-art/docker-diffusers-api/commit/ecd0b5d8efe734693ff9647cfc2d0bc0b8f90e42))

# 1.0.0 (2022-12-31)


### Bug Fixes

* **app:** clearPipelines() before loadModel() to free RAM ([ec45acf](https://github.com/kiri-art/docker-diffusers-api/commit/ec45acf7db7796682597d1d1c440d3742df84425))
* **app:** init: don't process MODEL_ID if not RUNTIME_DOWNLOADS ([683677f](https://github.com/kiri-art/docker-diffusers-api/commit/683677f0bdbd49c11cb0310c7c365047b536a4f7))
* **dockerfile:** bump diffusers to eb1abee693104dd45376dbddd614320f2a0beb24 ([1769330](https://github.com/kiri-art/docker-diffusers-api/commit/1769330d4ec1f5932591383daf078be0953accdc))
* **downloads:** model_url, model_id should be optional ([9a19e7e](https://github.com/kiri-art/docker-diffusers-api/commit/9a19e7e1e742c46471f9a7e6fcebacea5f887d35))
* **dreambooth:** don't crash on cleanup when no class_data_dir created ([36e64b1](https://github.com/kiri-art/docker-diffusers-api/commit/36e64b101bb12c7e09445f5958acaab1ab59a301))
* **dreambooth:** enable mixed_precision training, default to fp16 ([0430d23](https://github.com/kiri-art/docker-diffusers-api/commit/0430d2380b5c6e5e43f2c8657017ba701bfaec41))
* **gitScheduler:** fix deprecation warning s/from_config/from_pretrained/ ([92b2b43](https://github.com/kiri-art/docker-diffusers-api/commit/92b2b433bd9dfb4e1af1473cfa430e55bc83b170))
* **pipelines:** community pipelines, set torch_dtype too ([0cc1b63](https://github.com/kiri-art/docker-diffusers-api/commit/0cc1b63f72f98ad9267cdc71707bb4b533ad303d))
* **pipelines:** fix clearPipelines(), load model w/ correct precision ([3085412](https://github.com/kiri-art/docker-diffusers-api/commit/308541243c78cf528ebcd4c68900f5cdd52e6f8f))
* **requirements:** bumps transformers from 4.22.2 to 4.25.1 ([b13b58c](https://github.com/kiri-art/docker-diffusers-api/commit/b13b58c89fcd30e90ebb58c193c803450db43ebd))
* **s3:** incorrect value for tqdm causing crash ([9527ece](https://github.com/kiri-art/docker-diffusers-api/commit/9527ece90e4b5b4366f1c418d837dd659764203c))
* **send:** container_id detection, use /containers/ to grep ([5c0606a](https://github.com/kiri-art/docker-diffusers-api/commit/5c0606a0fdfd9b1a410b6f96eff009da6b768dbe))
* **tests:** default to DPMSolverMultistepScheduler and 20 steps ([a9c7bb0](https://github.com/kiri-art/docker-diffusers-api/commit/a9c7bb091821640a84d37d3090d365b7a54f2615))


### Features

* ability for custom config.yaml in CHECKPOINT_CONFIG_URL ([d2b507c](https://github.com/kiri-art/docker-diffusers-api/commit/d2b507ca225a033dda35897999e489541faecb8c))
* add PyPatchMatch for outpainting support ([3675bd3](https://github.com/kiri-art/docker-diffusers-api/commit/3675bd31a12d7b1f9627e34f59b661ea7261c272))
* **app:** don't track downloads in mem, check on disk ([51729e2](https://github.com/kiri-art/docker-diffusers-api/commit/51729e21440e4f0721b73ea497ddd2136306f11d))
* **app:** runtime downloads with MODEL_URL ([7abc4ac](https://github.com/kiri-art/docker-diffusers-api/commit/7abc4aced15f4aec441d4c220f39e046d2e35179))
* **app:** runtime downloads, re-use loaded model if requested again ([b84e822](https://github.com/kiri-art/docker-diffusers-api/commit/b84e822cacdb249693a301eb62a600ac9e0ee8f9))
* **callInputs:** `MODEL_ID`, `PIPELINE`, `SCHEDULER` now optional ([ef420a1](https://github.com/kiri-art/docker-diffusers-api/commit/ef420a1022b3d80950e7df79f1aff006e775c313))
* **cloud_cache:** normalize model_id and include precision ([ad1b2ef](https://github.com/kiri-art/docker-diffusers-api/commit/ad1b2efc60216c7a8854139ae816d78f6c4a9a19))
* **diffusers:** bump to v0.10.12 and one commit after (6b68afd) ([ec9117b](https://github.com/kiri-art/docker-diffusers-api/commit/ec9117b747985b7b3d80a4211c4e7bf6253a24a1))
* **diffusers:** bump to v0.9.0 ([0504d97](https://github.com/kiri-art/docker-diffusers-api/commit/0504d97e38eb85924ef7453c3c8690428f54870d))
* **docker:** diffusers-api-base image, build, run.sh ([1cbfc4f](https://github.com/kiri-art/docker-diffusers-api/commit/1cbfc4f41b46ea8d38600ac6902cf5f095357344))
* **dockerfile:** FROM_IMAGE build-arg to pick base image ([a0c37a6](https://github.com/kiri-art/docker-diffusers-api/commit/a0c37a6a87b300771f6ecf168b8bb1516caa5ab9))
* **Dockerfile:** make SDv2 the default (+ some formatting cleanup) ([c1e73ef](https://github.com/kiri-art/docker-diffusers-api/commit/c1e73efcdb6e5c95d36c83f9d1398182a1b7e77e))
* **dockerfile:** runtime downloads ([b40ae86](https://github.com/kiri-art/docker-diffusers-api/commit/b40ae868ce59ddb0232bcdb27ebb0a2c91068f51))
* **Dockerfile:** SAFETENSORS_FAST_GPU ([62209be](https://github.com/kiri-art/docker-diffusers-api/commit/62209be9963f9699ba32ea7520a361545b55034e))
* **download:** default_path as normalized_model_id.tar.zst ([5ad0d88](https://github.com/kiri-art/docker-diffusers-api/commit/5ad0d88b0b9b5a5a07596457c3bc83b7b32b25f5))
* **download:** delete .zst file after uncompress ([ab25280](https://github.com/kiri-art/docker-diffusers-api/commit/ab25280125bc1ccc38a0a2588fc09e33a576f6b0))
* **download:** record download timings ([7457e50](https://github.com/kiri-art/docker-diffusers-api/commit/7457e505c826c44d9f45a05fe486e819d442b4ca))
* **downloads:** runtime checkpoint conversion ([2414cd9](https://github.com/kiri-art/docker-diffusers-api/commit/2414cd9e3ac232273a1f2441134c65c25d0f7b49))
* **dreambooth:** save in safetensors format, tar up with -v ([5c3e86a](https://github.com/kiri-art/docker-diffusers-api/commit/5c3e86a8f99331c41c34b36c932b70e11f7b80b0))
* **errors:** try...catch everything, return as JSON ([901679c](https://github.com/kiri-art/docker-diffusers-api/commit/901679c7829796dc585af25f658cd6ab9115c7e7))
* **getScheduler:** make DPMSolverMultistepScheduler the default ([085d06f](https://github.com/kiri-art/docker-diffusers-api/commit/085d06f6b993a24b16521a1c3ee77d92289e04ed))
* **k-diffusion:** add pip package for use in k-diffusion shedulers ([3e901ad](https://github.com/kiri-art/docker-diffusers-api/commit/3e901adc64f750f5501b5dd19d87d0a5e294de22))
* **models:** store in ~/.cache/diffusers-api (volume support) ([8032ec1](https://github.com/kiri-art/docker-diffusers-api/commit/8032ec11b8f6590015110c9b89437f5619f2374c))
* **pipelines:** allow calling of ALL PIPELINES (official+community) ([1ccbaad](https://github.com/kiri-art/docker-diffusers-api/commit/1ccbaad1f405b8e5d16ca1a9880cc1d279f6d3f9))
* **pipelines:** initial community pipeline support ([7af45cf](https://github.com/kiri-art/docker-diffusers-api/commit/7af45cfdc4cbcc95c905834628775d0e8858509e))
* **s3:** s3client(), file_exists() methods ([0308af9](https://github.com/kiri-art/docker-diffusers-api/commit/0308af910d07be6d912104663263663b086def9c))
* **s3:** upload/download progress indicators ([76dd303](https://github.com/kiri-art/docker-diffusers-api/commit/76dd303a58a57b90ecc2c0038547b23b906ecca5))
* **send:** prefer env var CONTAINER_ID if set to full docker uuid ([eec5112](https://github.com/kiri-art/docker-diffusers-api/commit/eec511252035b8205f5365f45abb5777c164cb57))
* **send:** SEND_URL and SIGN_KEY now settable with build-vars ([01cf354](https://github.com/kiri-art/docker-diffusers-api/commit/01cf35461c5855a75651a30e3aeccb4ad1e9c8ac))
* **test:** allow TEST_URL to override https://localhost:8000/ ([9b46387](https://github.com/kiri-art/docker-diffusers-api/commit/9b463872257c0a3ffae553765aed62a2df6af717))
* **tests:** allow override BANANA_API_URL ([aca6aca](https://github.com/kiri-art/docker-diffusers-api/commit/aca6aca6e7ed46d0bf711548cea82a588fdd7d2a))

# CHANGELOG

* **NEXT MAIN**

  * Callinputs `MODEL_ID`, `PIPELINE` and `SCHEDULER` are **now optional**.
    If not specified, the default will be used, and returned in a `$meta`
    key in the result.

  * Tests: 1) Don't specify above defaults where possible, 2) Log exact
    inputs sent to container, 3) Log the full result sent back,
    substituting base64 image strings with their info, 4) format stack
    traces on caught errors from container.

* **NEXT MAIN (and already posted to forum)**

  * **Latest diffusers, SDv2.1**.  All the latest goodness, and upgraded some
    dependencies too.  Models are:

    * `stabilityai/stable-diffusion-2-1-base` (512x512)
    * `stabilityai/stable-diffusion-2-1` (768x768)

  * **ALL THE PIPELINES**.  We no longer load a list of hard-coded pipelines
    in `init()`.  Instead, we init and cache each on first use (for faster
    first calls on cold boots), and, *all* pipelines, both official diffusers
    and community pipelines, are available.
    [Full details](https://banana-forums.dev/t/all-your-pipelines-are-belong-to-us/83)

  * Dreambooth: Enable `mixed_precision` training, default to fp16.

  * [Experimental] **[Runtime downloads](https://banana-forums.dev/t/runtime-downloads-dont-download-during-build/81/3)** (Dreambooth
  only for now, more on the way)

  * **S3**: Add upload/download progress indicators.

  * Stable Diffusion has standardized **`image` instead of `init_image`** for
    all pipelines.  Using `init_image` now shows a deprecation warning and
    will be removed in future.

  * **Changed `sd-base` to `diffusers-api`** as the default tag / name used
    in the README examples and optional [./build][build script].

  * **Much better error handling**.  We now `try...except` both the pipeline
    run and entire `inference()` call, which will save you a trip to banana's
    logs which don't always even show these errors and sometimes just leave
    you with an unexplained stuck instance.  These kinds of errors are almost
    always a result of problematic callInputs and modelInputs used for the
    pipeline call, so finding them will be a lot easier now.

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
