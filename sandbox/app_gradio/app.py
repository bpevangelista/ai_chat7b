import gradio as gr
import random
import time

import json
import boto3

client = boto3.client("sagemaker-runtime")
chat_history = []

def invoke_llm_generation(message, chat_history):
    payload = {
        "message": message,
        "chat_history": "\n".join(chat_history),
        "parameters": {
        }
    }
    response = client.invoke_endpoint(
        EndpointName="pygmalion-6b-sep01-endpoint",
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    body = response["Body"].read()

    blob = json.loads(body)
    reply = blob["reply"]
    new_history_entry = blob["new_history_entry"]
    return reply, new_history_entry

def first_message():
    reply, new_history_entry = invoke_llm_generation("", [])
    return reply

def send_message(history, message):
    return [history + [(message, None)], ""]

def send_streaming(history):
    message = history[-1][0]
    history[-1][1] = ""

    reply, new_history_entry = invoke_llm_generation(message, chat_history)
    chat_history.append(new_history_entry)

    # store up to 7 interactions
    if len(chat_history) > 7:
        chat_history.pop(0)

    history[-1][1] = reply
    return history

custom_css = """
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

greetings = first_message()
with gr.Blocks(css=custom_css) as app:
    chatbot = gr.Chatbot(
        value=[(None, greetings)],
        bubble_full_width=False,
    )
    with gr.Row():
        textbox = gr.Textbox(show_label=False, container=False, scale=3)
        send = gr.Button("Send", scale=1)

    # funcion, inputs components, outputs components
    send.click(send_message, [chatbot, textbox], [chatbot, textbox], queue=False).then(
        send_streaming, chatbot, chatbot
    )

    textbox.submit(send_message, [chatbot, textbox], [chatbot, textbox], queue=False).then(
        send_streaming, chatbot, chatbot
    )

app.queue()
app.launch(share=True)