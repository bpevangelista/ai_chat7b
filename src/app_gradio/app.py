import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gradio as gr
import requests

CHAT_ACCESS_KEY = os.environ.get("CHAT_ACCESS_KEY")
CHAT_ENDPOINT_URI = os.environ.get("CHAT_ENDPOINT_URI")
MULTI_PERSONAS_ENABLED = True

DEFAULT_PERSONAS = [
    "professor_willow_en",
    "neighbor_justin_en",
    "yuki_hinashi_en",
    "yuki_hinashi_pt",
    "custom1",
    "custom2",
]
DEFAULT_PERSONA_STR = "professor_willow_en"

custom_css = """
div[data-testid="block-label"] {
    display: none;
}

.block {
    border-style: none;
}

.icon-button {
    display: none;
}

.wrap {
    width: none;
}

#chatbot {
    background-image: url('https://huggingface.co/spaces/bevangelista/llm_prof_willow/resolve/main/professor_willow_mj_s.png');
    background-size: 600px;
    background-repeat: no-repeat;
    background-position: center top;

    border-radius: 20px;
    flex-grow: 1; overflow: auto;
    overflow: hidden;
}


#component-0 {
    height: 600px;
    width: 600px;
}

.message {
    padding-bottom: 2px !important;
    padding-top: 2px !important;
}

.message-wrap {
    gap: var(--spacing-xl) !important;
    opacity: 0.87 !important;
}

.user-row {
    padding-top: 12px !important;
}
"""


def invoke_llm_generation(prompt: str, chat_history: list[str], persona_id: str):
    if not CHAT_ENDPOINT_URI:
        return None

    headers = {
        'Authorization': f'Bearer {CHAT_ACCESS_KEY}',
    }
    payload = {
        "prompt": prompt,
        "chat_history": chat_history,
        "persona_id": persona_id,
    }

    response = requests.post(CHAT_ENDPOINT_URI, headers=headers, json=payload)
    if response.status_code == 200:
        blob = response.json()
        reply = blob["reply"]
        chat_history.append(prompt)
        chat_history.append(reply)
        return reply
    return None


def get_first_message(chat_history: list[str], persona_id: str):
    chat_history.clear()
    reply = "This bot is currently offline"
    try:
        result = invoke_llm_generation("", [], persona_id)
        if result:
            reply = result[0] # TODO FIX show reply[0], 1s then reply[1]
    except:
        pass

    return [(None, reply)]


def send_message(history, message):
    return [history + [(message, None)], ""]


def send_streaming(history, chat_history, persona_id):
    message = history[-1][0]
    history[-1][1] = ""

    reply = invoke_llm_generation(message, chat_history, persona_id)

    # store up to 16 interactions
    if len(chat_history) > 16:
        chat_history.pop(0)

    history[-1][1] = reply
    return history


with gr.Blocks(css=custom_css) as app:
    if MULTI_PERSONAS_ENABLED:
        personas = DEFAULT_PERSONAS
        with gr.Row():
            persona = gr.Dropdown(personas, value=DEFAULT_PERSONA_STR, show_label=False, container=False, scale=3)
            reload = gr.Button("Reload Persona", scale=1)
    else:
        persona = gr.Textbox(value=DEFAULT_PERSONA_STR, visible=False)

    chat_history = gr.State([])
    chatbot = gr.Chatbot(
        elem_id='chatbot',
        bubble_full_width=False,
    )

    with gr.Row():
        textbox = gr.Textbox(show_label=False, container=False, scale=3)
        send = gr.Button("Send", scale=1)

    # fn, input components, output components
    send.click(send_message, [chatbot, textbox], [chatbot, textbox], queue=False).then(
        send_streaming, [chatbot, chat_history, persona], chatbot
    )

    if MULTI_PERSONAS_ENABLED:
        reload.click(lambda: [None, None], None, [chatbot, textbox], queue=False).then(
            get_first_message, [chat_history, persona], chatbot
        )

    textbox.submit(send_message, [chatbot, textbox], [chatbot, textbox], queue=False).then(
        send_streaming, [chatbot, chat_history, persona], chatbot
    )

    app.load(get_first_message, [chat_history, persona], chatbot)

app.queue()  # web-sockets
app.launch()
