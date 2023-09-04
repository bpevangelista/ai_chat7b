import gradio as gr
import json
import boto3

client = boto3.client("sagemaker-runtime")
chat_history = []

def first_message():
    return send_message("", [])

def send_message(message, chat_history):
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
    #print(body)

    blob = json.loads(body)
    reply = blob["reply"]
    new_history_entry = blob["new_history_entry"]
    return reply, new_history_entry

def predict(message, _):
    reply, new_history_entry = send_message(message, chat_history)
    chat_history.append(new_history_entry)

    # store up to 7 interactions
    if len(chat_history) > 7:
        chat_history.pop(0)

    return reply

textbox = gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7)
greetings, _ = first_message()
submit_btn=gr.Button("Send")
chatbot = gr.Chatbot(
    [(None, greetings)],
)

app = gr.ChatInterface(
    predict,
    retry_btn=None,
    undo_btn=None,
    clear_btn=None,
    chatbot=chatbot,
    submit_btn=submit_btn,
)

app.launch()

