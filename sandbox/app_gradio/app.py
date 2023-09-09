import os, sys
# web serving required
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gradio as gr
import json
import boto3, botocore

config = botocore.config.Config(
    read_timeout=20,
    retries={
        "max_attempts": 0
    }
)

# defined on huggingface - private env variables
region_name=os.environ.get("AWS_REGION")
aws_access_key_id=os.environ.get("AWS_ACCESS_KEY")
aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_sagemaker_endpoint_name=os.environ.get("AWS_SAGEMAKER_ENDPOINT_NAME")

# local
if not "HF_ENDPOINT" in os.environ:
    if len(sys.argv) == 2:
        aws_sagemaker_endpoint_name = sys.argv[1]
    else:
        print("  usage python3 app.py aws_sagemaker_endpoint_name")
        exit(1)

client = boto3.client(
    "sagemaker-runtime",
    config=config,
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

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

def invoke_llm_generation(message, chat_history, persona_id):
    payload = {
        "message": message,
        "chat_history": "\n".join(chat_history),
        "persona_id": persona_id,
        "parameters": {
        }
    }
    response = client.invoke_endpoint(
        EndpointName=aws_sagemaker_endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    body = response["Body"].read()
    blob = json.loads(body)
    reply = blob["reply"]
    new_history_entry = blob["new_history_entry"]
    return reply, new_history_entry

def get_first_message(persona_id):
    reply = "This bot is currently offline"
    try:
        reply, new_history_entry = invoke_llm_generation("", [], persona_id)
    except:
        pass
    return [(None, reply)]

def send_message(history, message):
    return [history + [(message, None)], ""]

def send_streaming(history, chat_history, persona_id):
    message = history[-1][0]
    history[-1][1] = ""

    reply, new_history_entry = invoke_llm_generation(message, chat_history, persona_id)
    chat_history.append(new_history_entry)

    # store up to 7 interactions
    if len(chat_history) > 7:
        chat_history.pop(0)

    history[-1][1] = reply
    return history

with gr.Blocks(css=custom_css) as app:
    personas = [
        "yuki_hinashi_en",
        "yuki_hinashi_pt",
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

    reload.click(lambda:[None, None], None, [chatbot, textbox], queue=False).then(
        get_first_message, persona, chatbot
    )

    app.load(get_first_message, persona, chatbot)

app.queue() # web-sockets
app.launch()
