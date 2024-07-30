"""
The gradio demo server for chatting with a single model.
"""

import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import os
import random
import time
import uuid

import gradio as gr
import requests

from fastchat.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    RATE_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
)
from fastchat.model.model_adapter import (
    get_conversation_template,
)
from fastchat.model.model_registry import get_model_info, model_info
from fastchat.serve.api_provider import get_api_provider_stream_iter
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    moderation_filter,
    parse_gradio_auth_creds,
    load_image,
)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "FastChat Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True, visible=True)
disable_btn = gr.Button(interactive=False)
invisible_btn = gr.Button(interactive=False, visible=False)
enable_text = gr.Textbox(
    interactive=True, visible=True, placeholder="👉 Enter your prompt and press ENTER"
)
disable_text = gr.Textbox(
    interactive=False,
    visible=True,
    placeholder='Press "🎲 New Round" to start over👇 (Note: Your vote shapes the leaderboard, please vote RESPONSIBLY!)',
)

controller_url = None
enable_moderation = False
use_remote_storage = False


# block_css = """
# .prose {
#     font-size: 105% !important;
# }

# #arena_leaderboard_dataframe table {
#     font-size: 105%;
# }
# #full_leaderboard_dataframe table {
#     font-size: 105%;
# }

# .tab-nav button {
#     font-size: 18px;
# }

# .chatbot h1 {
#     font-size: 130%;
# }
# .chatbot h2 {
#     font-size: 120%;
# }
# .chatbot h3 {
#     font-size: 110%;
# }

# #chatbot .prose {
#     font-size: 90% !important;
# }

# .sponsor-image-about img {
#     margin: 0 20px;
#     margin-top: 20px;
#     height: 40px;
#     max-height: 100%;
#     width: auto;
#     float: left;
# }

# .cursor {
#     display: inline-block;
#     width: 7px;
#     height: 1em;
#     background-color: black;
#     vertical-align: middle;
#     animation: blink 1s infinite;
# }

# .dark .cursor {
#     display: inline-block;
#     width: 7px;
#     height: 1em;
#     background-color: white;
#     vertical-align: middle;
#     animation: blink 1s infinite;
# }

# @keyframes blink {
#     0%, 50% { opacity: 1; }
#     50.1%, 100% { opacity: 0; }
# }

# .app {
#   max-width: 100% !important;
#   padding-left: 5% !important;
#   padding-right: 5% !important;
# }

# a {
#     color: #1976D2; /* Your current link color, a shade of blue */
#     text-decoration: none; /* Removes underline from links */
# }
# a:hover {
#     color: #63A4FF; /* This can be any color you choose for hover */
#     text-decoration: underline; /* Adds underline on hover */
# }
# """


# acknowledgment_md = """
# ### Terms of Service

# Users are required to agree to the following terms before using the service:

# The service is a research preview. It only provides limited safety measures and may generate offensive content.
# It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
# Please do not upload any private information.
# The service collects user dialogue data, including both text and images, and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) or a similar license.

# #### Please report any bug or issue to our [Discord](https://discord.gg/HSWAKCrnFx)/arena-feedback.

# ### Acknowledgment
# We thank [UC Berkeley SkyLab](https://sky.cs.berkeley.edu/), [Kaggle](https://www.kaggle.com/), [MBZUAI](https://mbzuai.ac.ae/), [a16z](https://www.a16z.com/), [Together AI](https://www.together.ai/), [Hyperbolic](https://hyperbolic.xyz/), [RunPod](https://runpod.io), [Anyscale](https://www.anyscale.com/), [HuggingFace](https://huggingface.co/) for their generous [sponsorship](https://lmsys.org/donations/).

# <div class="sponsor-image-about">
#     <img src="https://storage.googleapis.com/public-arena-asset/skylab.png" alt="SkyLab">
#     <img src="https://storage.googleapis.com/public-arena-asset/kaggle.png" alt="Kaggle">
#     <img src="https://storage.googleapis.com/public-arena-asset/mbzuai.jpeg" alt="MBZUAI">
#     <img src="https://storage.googleapis.com/public-arena-asset/a16z.jpeg" alt="a16z">
#     <img src="https://storage.googleapis.com/public-arena-asset/together.png" alt="Together AI">
#     <img src="https://storage.googleapis.com/public-arena-asset/hyperbolic_logo.png" alt="Hyperbolic">
#     <img src="https://storage.googleapis.com/public-arena-asset/runpod-logo.jpg" alt="RunPod">
#     <img src="https://storage.googleapis.com/public-arena-asset/anyscale.png" alt="AnyScale">
#     <img src="https://storage.googleapis.com/public-arena-asset/huggingface.png" alt="HuggingFace">
# </div>
# """

api_endpoint_info = {}


class State:
    def __init__(self, model_name="", is_vision=False):
        self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name
        self.oai_thread_id = None
        self.is_vision = is_vision
        self.mode = "Chat Mode"  # Default mode

        # NOTE(chris): This could be sort of a hack since it assumes the user only uploads one image. If they can upload multiple, we should store a list of image hashes.
        self.has_csam_image = False

        self.regen_support = True
        if "browsing" in model_name:
            self.regen_support = False
        self.init_system_prompt(self.conv, is_vision)

    def init_system_prompt(self, conv, is_vision):
        system_prompt = conv.get_system_message(is_vision)
        if len(system_prompt) == 0:
            return
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        system_prompt = system_prompt.replace("{{currentDateTime}}", current_date)
        conv.set_system_message(system_prompt)

    def to_gradio_chatbot(self):
        return self.conv.to_gradio_chatbot()

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
            }
        )
        if self.is_vision:
            base.update({"has_csam_image": self.has_csam_image})
        return base


def set_global_vars(controller_url_, enable_moderation_, use_remote_storage_):
    global controller_url, enable_moderation, use_remote_storage
    controller_url = controller_url_
    enable_moderation = enable_moderation_
    use_remote_storage = use_remote_storage_


def get_conv_log_filename(is_vision=False, has_csam_image=False):
    t = datetime.datetime.now()
    conv_log_filename = f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json"
    if is_vision and not has_csam_image:
        name = os.path.join(LOGDIR, f"vision-tmp-{conv_log_filename}")
    elif is_vision and has_csam_image:
        name = os.path.join(LOGDIR, f"vision-csam-{conv_log_filename}")
    else:
        name = os.path.join(LOGDIR, conv_log_filename)

    return name


def get_model_list(controller_url, register_api_endpoint_file, vision_arena):
    global api_endpoint_info

    # Add models from the controller
    if controller_url:
        ret = requests.post(controller_url + "/refresh_all_workers")
        assert ret.status_code == 200

        if vision_arena:
            ret = requests.post(controller_url + "/list_multimodal_models")
            models = ret.json()["models"]
        else:
            ret = requests.post(controller_url + "/list_language_models")
            models = ret.json()["models"]
    else:
        models = []

    # Add models from the API providers
    if register_api_endpoint_file:
        api_endpoint_info = json.load(open(register_api_endpoint_file))
        for mdl, mdl_dict in api_endpoint_info.items():
            mdl_vision = mdl_dict.get("vision-arena", False)
            mdl_text = mdl_dict.get("text-arena", True)
            if vision_arena and mdl_vision:
                models.append(mdl)
            if not vision_arena and mdl_text:
                models.append(mdl)

    # Remove anonymous models
    models = list(set(models))
    visible_models = models.copy()
    for mdl in models:
        if mdl not in api_endpoint_info:
            continue
        mdl_dict = api_endpoint_info[mdl]
        if mdl_dict["anony_only"]:
            visible_models.remove(mdl)

    # Sort models and add descriptions
    priority = {k: f"___{i:03d}" for i, k in enumerate(model_info)}
    models.sort(key=lambda x: priority.get(x, x))
    visible_models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"All models: {models}")
    logger.info(f"Visible models: {visible_models}")
    return visible_models, models


def load_demo_single(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown(choices=models, value=selected_model, visible=True)
    state = None
    return state, dropdown_update


def load_demo(url_params, request: gr.Request):
    global models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")

    if args.model_list_mode == "reload":
        models, all_models = get_model_list(
            controller_url, args.register_api_endpoint_file, vision_arena=False
        )

    return load_demo_single(models, url_params)


# def vote_last_response(state, vote_type, model_selector, request: gr.Request):
#     filename = get_conv_log_filename()
#     if "llava" in model_selector:
#         filename = filename.replace("2024", "vision-tmp-2024")

#     with open(filename, "a") as fout:
#         data = {
#             "tstamp": round(time.time(), 4),
#             "type": vote_type,
#             "model": model_selector,
#             "state": state.dict(),
#             "ip": get_ip(request),
#         }
#         fout.write(json.dumps(data) + "\n")
#     get_remote_logger().log(data)


# def upvote_last_response(state, model_selector, request: gr.Request):
#     ip = get_ip(request)
#     logger.info(f"upvote. ip: {ip}")
#     vote_last_response(state, "upvote", model_selector, request)
#     return ("",) + (disable_btn,) * 3


# def downvote_last_response(state, model_selector, request: gr.Request):
#     ip = get_ip(request)
#     logger.info(f"downvote. ip: {ip}")
#     vote_last_response(state, "downvote", model_selector, request)
#     return ("",) + (disable_btn,) * 3


# def flag_last_response(state, model_selector, request: gr.Request):
#     ip = get_ip(request)
#     logger.info(f"flag. ip: {ip}")
#     vote_last_response(state, "flag", model_selector, request)
#     return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"regenerate. ip: {ip}")
    if not state.regen_support:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 9
    state.conv.update_last_message(None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9


### Our optional button function in here ### 
# def add_text_start(state, model_selector, request: gr.Request):
#     ip = get_ip(request)
#     logger.info(f"add_text_start. ip: {ip}")
#     if state is None:
#         state = State(model_selector)
#     text = f"Hi, Doctor!"
#     state.conv.append_message(state.conv.roles[0], text)
#     state.conv.append_message(state.conv.roles[1], None)
#     return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9

def add_suggestion1(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_suggestion1. ip: {ip}")
    if state is None:
        state = State(model_selector)
    text = f"Based on suggestion 1, can you tell me more detail ?"
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9

def add_suggestion2(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_suggestion2. ip: {ip}")
    if state is None:
        state = State(model_selector)
    text = f"Based on suggestion 2, can you tell me more detail ?"
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9

def add_suggestion3(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_suggestion3. ip: {ip}")
    if state is None:
        state = State(model_selector)
    text = f"Based on suggestion 3, can you tell me more detail ?"
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9

def add_suggestion4(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_suggestion4. ip: {ip}")
    if state is None:
        state = State(model_selector)
    text = f"Based on suggestion 4, can you tell me more detail ?"
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9

# def symptom_func(state, model_selector, request: gr.Request):
#     ip = get_ip(request)
#     logger.info(f"symptom_func. ip: {ip}")
#     if state is None:
#         state = State(model_selector)
#     text = f"Doctor, What symptoms might I be experiencing ?"
#     state.conv.append_message(state.conv.roles[0], text)
#     state.conv.append_message(state.conv.roles[1], None)
#     return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9

# def clinic_func(state, model_selector, request: gr.Request):
#     ip = get_ip(request)
#     logger.info(f"clinic_func. ip: {ip}")
#     if state is None:
#         state = State(model_selector)
#     text = f"Doctor, What specialist should I go ?"
#     state.conv.append_message(state.conv.roles[0], text)
#     state.conv.append_message(state.conv.roles[1], None)
#     return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9

def switch_mode(state, mode, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"switch_mode. ip: {ip}")
    if state is None:
        state = State(model_selector)
    if mode == "Urgent Mode":
        state.mode = "Urgent Mode"
    elif mode == "Chat Mode":
        state.mode = "Chat Mode"
    else:
        state.mode = "Chat Mode"  # Default to Chat Mode if something goes wrong
    return state  # Return the updated mode for display

def clear_history(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"clear_history. ip: {ip}")
    if state is None:
        state = State(model_selector)
    else:
        original_mode = state.mode
        state = State(model_selector)
        state.mode = original_mode
    return (state, [], "") + (disable_btn,) * 9

####################################################   


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    elif "x-forwarded-for" in request.headers:
        ip = request.headers["x-forwarded-for"]
        if "," in ip:
            ip = ip.split(",")[0]
    else:
        ip = request.client.host
    return ip


def add_text(state, model_selector, text, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 9

    all_conv_text = state.conv.get_prompt()
    all_conv_text = all_conv_text[-2000:] + "\nuser: " + text
    flagged = moderation_filter(all_conv_text, [state.model_name])
    # flagged = moderation_filter(text, [state.model_name])
    if flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    if (len(state.conv.messages) - state.conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (no_change_btn,) * 9

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    if(state.mode == "Chat Mode"): 
        text = text
        #print(text)
    else: 
        text = "Doctor, the situation is urgent. The patient has the following symptoms: " + text + "\nWhat urgent measures should be taken?"
    
    # text = "Doctor, " + text + " Based on my symptoms, please give me 4 suggestions."
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9

def add_text_suggest(state, model_selector, text, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_text_suggest. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    all_conv_text = state.conv.get_prompt()
    all_conv_text = all_conv_text[-2000:] + "\nuser: " + text
    flagged = moderation_filter(all_conv_text, [state.model_name])
    # flagged = moderation_filter(text, [state.model_name])
    if flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    if (len(state.conv.messages) - state.conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (no_change_btn,) * 9

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    text = text + " Based on my symptoms, please list 4 suggestions for me."
    
    # text = "Doctor, " + text + " Based on my symptoms, please give me 4 suggestions."
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9

def add_text_clinic(state, model_selector, text, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_text_clinic. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    all_conv_text = state.conv.get_prompt()
    all_conv_text = all_conv_text[-2000:] + "\nuser: " + text
    flagged = moderation_filter(all_conv_text, [state.model_name])
    # flagged = moderation_filter(text, [state.model_name])
    if flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    if (len(state.conv.messages) - state.conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (no_change_btn,) * 9

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    text = text + " Which type of specialist should i see?"
    # text = "Doctor, " + text + " What specialist should I go ?"
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9


def add_text_symptom(state, model_selector, text, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_text_symptom. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    all_conv_text = state.conv.get_prompt()
    all_conv_text = all_conv_text[-2000:] + "\nuser: " + text
    flagged = moderation_filter(all_conv_text, [state.model_name])
    # flagged = moderation_filter(text, [state.model_name])
    if flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    if (len(state.conv.messages) - state.conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (no_change_btn,) * 9

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    text = text + " What symptoms might I be experiencing?"
    # text = "Doctor, " + text + " What symptoms might I be experiencing?"
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 9


def model_worker_stream_iter(
    conv,
    model_name,
    worker_addr,
    prompt,
    temperature,
    repetition_penalty,
    top_p,
    max_new_tokens,
    images,
):
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }

    logger.info(f"==== request ====\n{gen_params}")

    if len(images) > 0:
        gen_params["images"] = images

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def is_limit_reached(model_name, ip):
    monitor_url = "http://localhost:9090"
    try:
        ret = requests.get(
            f"{monitor_url}/is_limit_reached?model={model_name}&user_id={ip}", timeout=1
        )
        obj = ret.json()
        return obj
    except Exception as e:
        logger.info(f"monitor error: {e}")
        return None


def bot_response(
    state,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
    apply_rate_limit=True,
    use_recommended_config=False,
):
    ip = get_ip(request)
    logger.info(f"bot_response. ip: {ip}")
    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 9
        return

    if apply_rate_limit:
        ret = is_limit_reached(state.model_name, ip)
        if ret is not None and ret["is_limit_reached"]:
            error_msg = RATE_LIMIT_MSG + "\n\n" + ret["reason"]
            logger.info(f"rate limit reached. ip: {ip}. error_msg: {ret['reason']}")
            state.conv.update_last_message(error_msg)
            yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 9
            return

    conv, model_name = state.conv, state.model_name
    model_api_dict = (
        api_endpoint_info[model_name] if model_name in api_endpoint_info else None
    )
    images = conv.get_images()

    if model_api_dict is None:
        # Query worker address
        ret = requests.post(
            controller_url + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            conv.update_last_message(SERVER_ERROR_MSG)
            yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 9
            return

        # Construct prompt.
        # We need to call it here, so it will not be affected by "▌".
        prompt = conv.get_prompt()
        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        stream_iter = model_worker_stream_iter(
            conv,
            model_name,
            worker_addr,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            max_new_tokens,
            images,
        )
    else:
        # Remove system prompt for API-based models unless specified
        custom_system_prompt = model_api_dict.get("custom_system_prompt", False)
        if not custom_system_prompt:
            conv.set_system_message("")

        if use_recommended_config:
            recommended_config = model_api_dict.get("recommended_config", None)
            if recommended_config is not None:
                temperature = recommended_config.get("temperature", temperature)
                top_p = recommended_config.get("top_p", top_p)
                max_new_tokens = recommended_config.get(
                    "max_new_tokens", max_new_tokens
                )

        stream_iter = get_api_provider_stream_iter(
            conv,
            model_name,
            model_api_dict,
            temperature,
            top_p,
            max_new_tokens,
            state,
        )

    html_code = ' <span class="cursor"></span> '

    # conv.update_last_message("▌")
    conv.update_last_message(html_code)
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 9

    try:
        data = {"text": ""}
        for i, data in enumerate(stream_iter):
            if data["error_code"] == 0:
                output = data["text"].strip()
                conv.update_last_message(output + "▌")
                # conv.update_last_message(output + html_code)
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 9
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                conv.update_last_message(output)
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    disable_btn,
                    enable_btn,
                    enable_btn,
                )
                return
        output = data["text"].strip()
        conv.update_last_message(output)
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 9
    except requests.exceptions.RequestException as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return
    except Exception as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    finish_tstamp = time.time()
    logger.info(f"{output}")

    conv.save_new_images(
        has_csam_images=state.has_csam_image, use_remote_storage=use_remote_storage
    )

    filename = get_conv_log_filename(
        is_vision=state.is_vision, has_csam_image=state.has_csam_image
    )

    with open(filename, "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)

def get_model_description_md(models):
    model_description_md = """
| | | |
| ---- | ---- | ---- |
"""
    ct = 0
    visited = set()
    for i, name in enumerate(models):
        minfo = get_model_info(name)
        if minfo.simple_name in visited:
            continue
        visited.add(minfo.simple_name)
        one_model_md = f"[{minfo.simple_name}]({minfo.link}): {minfo.description}"

        if ct % 3 == 0:
            model_description_md += "|"
        model_description_md += f" {one_model_md} |"
        if ct % 3 == 2:
            model_description_md += "\n"
        ct += 1
    return model_description_md


def build_single_model_ui(models, add_promotion_links=False):

    state = gr.State()
    # state = State(models[0] if len(models) > 0 else "")
    
    title_html = f"""
    <h1 style="text-align: center; font-size: 260%;">️👨‍⚕️ ADAR-MED 👩‍⚕️</h1>
    <h3 style="text-align: center;">An AI-Driven Assistant for Rapid Medical Evaluation and Diagnosis</h3>
    <p style="text-align: center;">Links: <a href="https://www.hackster.io/contests/amd2023/hardware_applications/16954">Proposal</a> | <a href="https://www.hackster.io/519710/adar-med-ai-driven-assistant-for-rapid-medical-diagnosis-8f6e0c#toc-web-ui-4">Report</a> | <a href="https://github.com/kai-0430/ADAR-MED">GitHub</a><br /></p>
    """
    title = gr.HTML(title_html)

    
    with gr.Group(elem_id="chatbot_window"):
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",  # default value selected in dropdown
                interactive=True,
                label="Model",
            )
            model_description = gr.Textbox(
                value="medalpaca-7b is a large language model specifically fine-tuned for medical domain tasks. It is based on LLaMA and contains 7 billion parameters.",
                label="Description",
                interactive=False,
                elem_id="model_description_textbox",
                # placeholder=,
            )
            
        with gr.Accordion("Parameters", open=False) as parameter_row:
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.01,
                interactive=True,
                label="Temperature",
            )
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.01,
                interactive=True,
                label="Top P",
            )
            max_output_tokens = gr.Slider(
                minimum=16,
                maximum=2048,
                value=1024,
                step=64,
                interactive=True,
                label="Max output tokens",
            )            

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            label="Scroll down and start chatting",
            height=650,
            show_copy_button=True,
        )
        
    with gr.Group(elem_id="prompt_window"):
        with gr.Row():
            mode_select = gr.Dropdown(
                ["Chat Mode", "Urgent Mode"],
                label="Mode",
                value="Chat Mode",  # Default to Chat Mode
                scale=0
            )
            
            # @gr.render(inputs=mode_select)
            # def show_textbox(mode):
            #     if mode == 'Urgent Mode':
            #         placeholder_text = "👉 Enter your symptoms and press ENTER!"
            #     else:
            #         placeholder_text = "👉 Enter your prompt and press ENTER!"
            #     textbox = gr.Textbox(
            #         show_label=True,
            #         label="Text",
            #         placeholder=placeholder_text,
            #         elem_id="input_box",
            #     )
            textbox = gr.Textbox(
                show_label=True,
                label="Text",
                placeholder="👉 Enter your prompt and press ENTER!",
                elem_id="input_box",
            )
            send_btn = gr.Button(value="Send", variant="primary", scale=0)
        # Action buttons including suggestion buttons
        with gr.Row(variant='panel') as button_row:         
            symptom_btn = gr.Button(value="🤒 What symptoms may I have?")
            suggest_send_btn = gr.Button(value="📝 Could you give me some suggestions?")
            clinic_btn = gr.Button(value="🏨Which specialist should I see?")
        # suggestion buttons
        with gr.Row(variant='panel') as button_row:
            suggestion1_btn = gr.Button(value="🔶 Suggestion 1")
            suggestion2_btn = gr.Button(value="🔶 Suggestion 2")
            suggestion3_btn = gr.Button(value="🔶 Suggestion 3")
            suggestion4_btn = gr.Button(value="🔶 Suggestion 4")
            
    with gr.Row(variant='panel') as button_row:
        # upvote_btn = gr.Button(value="👍  Upvote")
        # downvote_btn = gr.Button(value="👎  Downvote")
        # flag_btn = gr.Button(value="⚠️  Flag")
        regenerate_btn = gr.Button(value="🔄  Regenerate", )
        clear_btn = gr.Button(value="🗑️  Clear history", )
            
    # Register listeners
    btn_list = [regenerate_btn, clear_btn, suggestion1_btn, suggestion2_btn, suggestion3_btn, suggestion4_btn, symptom_btn, suggest_send_btn, clinic_btn]
    
    # Handle the mode switch with the dropdown change
    mode_select.change(
            fn=switch_mode,  # Function to update the mode
            inputs=[state, mode_select, model_selector],  # Pass dropdown value and state
            outputs=state  # Update the mode display
    )
    
    textbox.submit(
        add_text,
        [state, model_selector, textbox],
        [state, chatbot, textbox] + btn_list,
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
       
    send_btn.click(
        add_text,
        [state, model_selector, textbox],
        [state, chatbot, textbox] + btn_list,
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
        
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    
    clear_btn.click(
        clear_history,
        [state, model_selector],
        [state, chatbot, textbox] + btn_list)

    model_selector.change(
        clear_history,
        [state, model_selector],
        [state, chatbot, textbox] + btn_list
    )

    suggestion1_btn.click(
        add_suggestion1,
        [state, model_selector],
        [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    
    suggestion2_btn.click(
        add_suggestion2,
        [state, model_selector],
        [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    
    suggestion3_btn.click(
        add_suggestion3,
        [state, model_selector],
        [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    
    suggestion4_btn.click(
        add_suggestion4,
        [state, model_selector],
        [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    
    suggest_send_btn.click(
        add_text_suggest,
        [state, model_selector, textbox],
        [state, chatbot, textbox] + btn_list,
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
        
    symptom_btn.click(
        add_text_symptom,
        [state, model_selector, textbox],
        [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    
    clinic_btn.click(
        add_text_clinic,
        [state, model_selector, textbox],
        [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )    
    
    # upvote_btn.click(
    #     upvote_last_response,
    #     [state, model_selector],
    #     [textbox, upvote_btn, downvote_btn, flag_btn],
    # )
    # downvote_btn.click(
    #     downvote_last_response,
    #     [state, model_selector],
    #     [textbox, upvote_btn, downvote_btn, flag_btn],
    # )
    # flag_btn.click(
    #     flag_last_response,
    #     [state, model_selector],
    #     [textbox, upvote_btn, downvote_btn, flag_btn],
    # )
   
    return [state, model_selector]


def build_demo(models):
    with gr.Blocks(
        title="Medical Advice ChatBot",
        theme=gr.themes.Default(),
        # css=block_css,
    ) as demo:
        url_params = gr.JSON(visible=False)

        state, model_selector = build_single_model_ui(models)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        if args.show_terms_of_use:
            load_js = get_window_url_params_with_tos_js
        else:
            load_js = get_window_url_params_js

        demo.load(
            load_demo,
            [url_params],
            [
                state,
                model_selector,
            ],
            js=load_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable content moderation to block unsafe inputs",
    )
    parser.add_argument(
        "--show-terms-of-use",
        action="store_true",
        help="Shows term of use before loading the demo",
    )
    parser.add_argument(
        "--register-api-endpoint-file",
        type=str,
        help="Register API-based model endpoints from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
    )
    parser.add_argument(
        "--gradio-root-path",
        type=str,
        help="Sets the gradio root path, eg /abc/def. Useful when running behind a reverse-proxy or at a custom URL path prefix",
    )
    parser.add_argument(
        "--use-remote-storage",
        action="store_true",
        default=False,
        help="Uploads image files to google cloud storage if set to true",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate, args.use_remote_storage)
    models, all_models = get_model_list(
        args.controller_url, args.register_api_endpoint_file, vision_arena=False
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(models)
    demo.queue(
        default_concurrency_limit=args.concurrency_count,
        status_update_rate=10,
        api_open=False,
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
        root_path=args.gradio_root_path,
    )
