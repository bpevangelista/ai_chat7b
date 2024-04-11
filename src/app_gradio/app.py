import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gradio as gr
import requests

CHAT_ACCESS_KEY = os.environ.get("CHAT_ACCESS_KEY")
CHAT_ENDPOINT_URI = os.environ.get("CHAT_ENDPOINT_URI")

custom_css = """
:root {
  --block-background-fill: transparent;
}

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

.main {
    background-image: url(https://media.discordapp.net/attachments/1210706446438768761/1210709974615851068/soij__Yuki_a_blond-haired_brown-eyed_girl_leans_against_a_wall__821d4d8d-2d1e-47a3-9662-6b7ac94f0f1b.png?ex=65eb8c87&is=65d91787&hm=50f3151529fd62c78bea0c657a7dcacf9ba7370d8fded75c58f1b262d28faf95&=&format=webp&quality=lossless&width=897&height=897);
    background-size: 400px 800px;
    width: 400px;
    height: 800px;
}

.message {
    padding-bottom: 2px !important;
    padding-top: 2px !important;
}

.message-wrap {
    gap: var(--spacing-xl) !important;
}

.user-row {
    padding-top: 12px !important;
}
"""


def invoke_llm_generation(prompt: str, chat_history: list[str], persona_id: str):
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


def get_first_message(chat_history, persona_id):
    chat_history.clear()
    reply = "This bot is currently offline"
    try:
        reply = invoke_llm_generation("", [], persona_id)
        reply = reply[0] # TODO FIX show reply[0], 1s then reply[1]
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
    personas = [
        "yuki_hinashi_en",
        "neighbor_justin_en",
        #"yuki_hinashi_pt",
        "custom1",
        "custom2",
    ]
    with gr.Row():
        persona = gr.Dropdown(personas, value=personas[0], show_label=False, container=False, scale=3)
        reload = gr.Button("Reload Persona", scale=1)

    chat_history = gr.State([])
    chatbot = gr.Chatbot(
        bubble_full_width=False,
    )

    with gr.Row():
        textbox = gr.Textbox(show_label=False, container=False, scale=3)
        send = gr.Button("Send", scale=1)

    # fn, input components, output components
    send.click(send_message, [chatbot, textbox], [chatbot, textbox], queue=False).then(
        send_streaming, [chatbot, chat_history, persona], chatbot
    )

    textbox.submit(send_message, [chatbot, textbox], [chatbot, textbox], queue=False).then(
        send_streaming, [chatbot, chat_history, persona], chatbot
    )

    reload.click(lambda: [None, None], None, [chatbot, textbox], queue=False).then(
        get_first_message, [chat_history, persona], chatbot
    )

    app.load(get_first_message, [chat_history, persona], chatbot)

app.queue()  # web-sockets
app.launch()
