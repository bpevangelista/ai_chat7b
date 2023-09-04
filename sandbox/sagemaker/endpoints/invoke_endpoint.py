import json
import boto3

client = boto3.client("sagemaker-runtime")

def send_message(message, chat_history):
    content_type = "application/json"
    payload = {
        "message": message,
        "chat_history": "\n".join(chat_history),
        "parameters": {
        }
    }
    response = client.invoke_endpoint(
        EndpointName="pygmalion-6b-sep01-endpoint", ContentType=content_type, Body=json.dumps(payload)
    )
    body = response["Body"].read()
    #print(body)

    blob = json.loads(body)
    reply = blob["reply"]
    new_history_entry = blob["new_history_entry"]
    return reply, new_history_entry

def main_loop():
    greeting, _ = send_message(message="", chat_history="")
    print(f"{greeting}\n")

    chat_history = []
    while True:
        prompt = input("You: ")
        if prompt == "":
            break

        reply, new_history_entry = send_message(prompt, chat_history)
        chat_history.append(new_history_entry)

        # store up to 6 interactions
        if len(chat_history) > 6:
            chat_history.pop(0)

        print(f"{reply}\n")

main_loop()