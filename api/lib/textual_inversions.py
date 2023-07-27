import json
import re
import os
from utils import Storage
from .vars import MODELS_DIR

last_textual_inversions = None
last_textual_inversion_model = None
loaded_textual_inversion_tokens = []

tokenRe = re.compile(
    r"[#&]{1}fname=(?P<fname>[^\.]+)\.(?:pt|safetensors)(&token=(?P<token>[^&]+))?$"
)


def strMap(str: str):
    match = re.search(tokenRe, str)
    print(match)
    if match:
        return match.group("token") or match.group("fname")


def extract_tokens_from_list(textual_inversions: list):
    return list(map(strMap, textual_inversions))


def handle_textual_inversions(textual_inversions: list, model):
    global last_textual_inversions
    global last_textual_inversion_model
    global loaded_textual_inversion_tokens

    textual_inversions_str = json.dumps(textual_inversions)
    if (
        textual_inversions_str is not last_textual_inversions
        or model is not last_textual_inversion_model
    ):
        if (model is not last_textual_inversion_model):
            loaded_textual_inversion_tokens = []
            last_textual_inversion_model = model
        # print({"textual_inversions": textual_inversions})
        # tokens_to_load = extract_tokens_from_list(textual_inversions)
        # print({"tokens_loaded": loaded_textual_inversion_tokens})
        # print({"tokens_to_load": tokens_to_load})
        #
        # for token in loaded_textual_inversion_tokens:
        #     if token not in tokens_to_load:
        #         print("[TextualInversion] Removing uneeded token: " + token)
        #         del pipeline.tokenizer.get_vocab()[token]
        #         # del pipeline.text_encoder.get_input_embeddings().weight.data[token]
        #         pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
        #
        # loaded_textual_inversion_tokens = tokens_to_load

        last_textual_inversions = textual_inversions_str
        for textual_inversion in textual_inversions:
            storage = Storage(textual_inversion, no_raise=True)
            if storage:
                storage_query_fname = storage.query.get("fname")
                if storage_query_fname:
                    fname = storage_query_fname[0]
                else:
                    fname = textual_inversion.split("/").pop()
                path = os.path.join(MODELS_DIR, "textual_inversion--" + fname)
                if not os.path.exists(path):
                    storage.download_file(path)
                print("Load textual inversion " + path)
                token = storage.query.get("token", None)
                if token not in loaded_textual_inversion_tokens:
                    model.load_textual_inversion(
                        path, token=token, local_files_only=True
                    )
                    loaded_textual_inversion_tokens.append(token)
            else:
                print("Load textual inversion " + textual_inversion)
                model.load_textual_inversion(textual_inversion)
