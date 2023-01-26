# Based on https://github.com/huggingface/diffusers/blob/946d1cb200a875f694818be37c9c9f7547e9db45/examples/dreambooth/train_dreambooth.py

# Reasons for not using that file directly:
#
#   1) Use our already loded model from `init()`
#   2) Callback to run after every iteration

# Deps

#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import argparse
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

# DDA
from send import send, get_now
from utils import Storage
import subprocess
import re
import shutil

# Our original code in docker-diffusers-api:

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")


def TrainDreamBooth(model_id: str, pipeline, model_inputs, call_inputs):
    # required inputs: instance_images instance_prompt

    params = {
        # Defaults
        "pretrained_model_name_or_path": model_id,  # DDA, TODO
        # Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be
        # float32 precision.
        "revision": None,
        "tokenizer_name": None,
        "instance_data_dir": "instance_data_dir",  # DDA TODO
        "class_data_dir": "class_data_dir",  # DDA, was: None,
        # instance_prompt
        "class_prompt": None,
        "with_prior_preservation": False,
        "prior_loss_weight": 1.0,
        "num_class_images": 100,
        "output_dir": "text-inversion-model",
        "seed": None,
        "resolution": 512,
        # Whether to center crop the input images to the resolution. If not set, the images will be randomly
        # cropped. The images will be resized to the resolution first before cropping.
        "center_crop": False,
        # Whether to train the text encoder. If set, the text encoder should be float32 precision.
        "train_text_encoder": None,
        "train_batch_size": 1,  # DDA, was: 4
        "sample_batch_size": 1,  # DDA, was: 4,
        "num_train_epochs": 1,
        "max_train_steps": 800,  # DDA, was: None,
        "checkpointing_steps": 1000000000,  # DDA, was: 500
        "resume_from_checkpoint": None,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,  # DDA was: None (needed for 16GB)
        "learning_rate": 5e-6,
        "scale_lr": False,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,  # DDA, was: 500,
        "lr_num_cycles": 1,
        # Power factor of the polynomial scheduler
        "lr_power": 1.0,
        "use_8bit_adam": True,  # DDA, was: None (needed for 16GB)
        # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
        "dataloader_num_workers": 0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-6,
        "adam_epsilon": 1e-08,
        "max_grad_norm": 1.0,
        "push_to_hub": None,
        "hub_token": HF_AUTH_TOKEN,
        "hub_model_id": None,
        "logging_dir": "logs",
        # Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        "allow_tf32": None,
        # The integration to report the results and logs to. Supported platforms are `"tensorboard"`
        # (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.
        "report_to": "tensorboard",
        "mixed_precision": "fp16",  # DDA, was: None
        # Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >=
        # 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32.
        "prior_generation_precision": None,  # "no", "fp32", "fp16", "bf16"
        "local_rank": -1,
        "enable_xformers_memory_efficient_attention": True,  # DDA, was: None
        # Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain
        # behaviors, so disable this argument if it causes any problems. More info:
        # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
        "set_grads_to_none": None,
    }

    instance_images = model_inputs["instance_images"]
    del model_inputs["instance_images"]

    params.update(model_inputs)
    print(model_inputs)

    args = argparse.Namespace(**params)
    print(args)

    result = {}

    if not args.push_to_hub and call_inputs.get("dest_url", None) == None:
        print()
        print("WARNING: Neither modelInputs.push_to_hub nor callInputs.dest_url")
        print("was given.  After training, your model won't be uploaded anywhere.")
        print()
        result.update({"no_upload": True})

    # TODO, not save at all... we're just getting it working
    # if its a hassle, in interim, at least save to unique dir
    if not os.path.exists(args.instance_data_dir):
        os.mkdir(args.instance_data_dir)
    for i, image in enumerate(instance_images):
        image.save(args.instance_data_dir + "/image" + str(i) + ".png")

    subprocess.run(["ls", "-l", args.instance_data_dir])

    result = result | main(args, pipeline)

    dest_url = call_inputs.get("dest_url")
    if dest_url:
        storage = Storage(dest_url)
        filename = storage.path if storage.path != "" else args.output_dir
        filename = filename.split("/").pop()
        print(filename)
        if not re.search(r"\.", filename):
            filename += ".tar.zstd"
        print(filename)

        # fp16 model timings: zip 1m20s, tar+zstd 4s and a tiny bit smaller!
        send("compress", "start", {})

        # TODO, steaming upload (turns out docker disk write is super slow)
        subprocess.run(
            f"tar cvf - -C {args.output_dir} . | zstd -o {filename}",
            shell=True,
            check=True,  # TODO, rather don't raise and return an error in JSON
        )

        send("compress", "done")
        subprocess.run(["ls", "-l", filename])

        send("upload", "start", {})
        upload_result = storage.upload_file(filename, filename)
        send("upload", "done")
        print(upload_result)
        os.remove(filename)

    # Cleanup
    shutil.rmtree(args.output_dir)
    shutil.rmtree(args.class_data_dir, ignore_errors=True)

    return result


# What follows is mostly the original train_dreambooth.py
# Any changes are marked with in comments with [DDA].

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.12.0")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args, init_pipeline):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            # DDA
            # torch_dtype = (
            #    torch.float16 if accelerator.device.type == "cuda" else torch.float32
            # )
            # if args.prior_generation_precision == "fp32":
            #     torch_dtype = torch.float32
            # elif args.prior_generation_precision == "fp16":
            #     torch_dtype = torch.float16
            # elif args.prior_generation_precision == "bf16":
            #     torch_dtype = torch.bfloat16
            # DDA
            pipeline = init_pipeline
            pipeline.safety_checker = None
            # pipeline = DiffusionPipeline.from_pretrained(
            #     args.pretrained_model_name_or_path,
            #     torch_dtype=torch_dtype,
            #     safety_checker=None,
            #     revision=args.revision,
            # )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            # pipeline.to(accelerator.device) # DDA already done

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            create_repo(
                repo_name,
                exist_ok=True,
                token=args.hub_token,
                private=True,  # DDA
            )
            repo = Repository(
                args.output_dir,
                clone_from=repo_name,
                token=args.hub_token,
                # use_auth_token=args.hub_token,  # DDA
            )

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, revision=args.revision, use_fast=False
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = init_pipeline.components["tokenizer"]  # DDA
        # tokenizer = AutoTokenizer.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     subfolder="tokenizer",
        #     revision=args.revision,
        #     use_auth_token=args.hub_token,  # DDA
        #     local_files_only=True,  # DDA
        # )

    # import correct text encoder class
    # DDA
    # text_encoder_cls = import_model_class_from_model_name_or_path(
    #     args.pretrained_model_name_or_path,
    #     args.revision
    # )

    # Load scheduler and models
    # noise_scheduler = DDPMScheduler.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="scheduler",
    #     use_auth_token=args.hub_token,  # DDA
    #     local_files_only=True,  # DDA
    # )

    # text_encoder = text_encoder_cls.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="text_encoder",
    #     revision=args.revision,
    #     use_auth_token=args.hub_token,  # DDA
    #     local_files_only=True,  # DDA
    # )
    # vae = AutoencoderKL.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="vae",
    #     revision=args.revision,
    #     use_auth_token=args.hub_token,  # DDA
    #     local_files_only=True,  # DDA
    # )
    # unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="unet",
    #     revision=args.revision,
    #     use_auth_token=args.hub_token,  # DDA
    #     local_files_only=True,  # DDA
    # )
    # print("pipeline.disable_xformers_memory_efficient_attention()")
    # init_pipeline.disable_xformers_memory_efficient_attention()
    noise_scheduler = init_pipeline.components["scheduler"]  # DDA
    text_encoder = init_pipeline.components["text_encoder"]  # DDA
    vae = init_pipeline.components["vae"]  # DDA
    unet = init_pipeline.components["unet"]  # DDA

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if (
        args.train_text_encoder
        and accelerator.unwrap_model(text_encoder).dtype != torch.float32
    ):
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters())
        if args.train_text_encoder
        else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # DDA
    send("training", "start", {})

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )

                    # Compute prior loss
                    prior_loss = F.mse_loss(
                        model_pred_prior.float(), target_prior.float(), reduction="mean"
                    )

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        pipeline.save_pretrained(save_path)
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    send("training", "done")  # DDA
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
            local_files_only=True,  # DDA
        )
        pipeline.save_pretrained(args.output_dir, safe_serialization=True)

        if args.push_to_hub:
            # DDA
            send("upload", "start", {})

            repo.push_to_hub(
                commit_message="End of training",
                # DDA need to think about this, quite nice to not block, then could
                # upload while training next request.  But, timeout will kill an unused
                # process...  what else?
                blocking=True,  # DDA, was: False,
                auto_lfs_prune=True,
            )

            # DDA
            send("upload", "done")

    accelerator.end_training()

    # DDA
    return {"done": True}
