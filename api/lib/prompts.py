from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType


def prepare_prompts(pipeline, model_inputs, is_sdxl):
    textual_inversion_manager = DiffusersTextualInversionManager(pipeline)
    if is_sdxl:
        compel = Compel(
            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
            # diffusers has no ti in sdxl yet
            # https://github.com/huggingface/diffusers/issues/4376#issuecomment-1659016141
            # textual_inversion_manager=textual_inversion_manager,
            truncate_long_prompts=False,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        conditioning, pooled = compel(model_inputs.get("prompt"))
        negative_conditioning, negative_pooled = compel(
            model_inputs.get("negative_prompt")
        )
        [
            conditioning,
            negative_conditioning,
        ] = compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning]
        )
        model_inputs.update(
            {
                "prompt": None,
                "negative_prompt": None,
                "prompt_embeds": conditioning,
                "negative_prompt_embeds": negative_conditioning,
                "pooled_prompt_embeds": pooled,
                "negative_pooled_prompt_embeds": negative_pooled,
            }
        )

    else:
        compel = Compel(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            textual_inversion_manager=textual_inversion_manager,
            truncate_long_prompts=False,
        )
        conditioning = compel(model_inputs.get("prompt"))
        negative_conditioning = compel(model_inputs.get("negative_prompt"))
        [
            conditioning,
            negative_conditioning,
        ] = compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning]
        )
        model_inputs.update(
            {
                "prompt": None,
                "negative_prompt": None,
                "prompt_embeds": conditioning,
                "negative_prompt_embeds": negative_conditioning,
            }
        )
