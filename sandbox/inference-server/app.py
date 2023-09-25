from flask import Flask
import logging, socket

from inference import InferenceEngine, InferenceModel, ChatPersonas, DefaultLogger

app = Flask(__name__)

#option_a = os.getenv('OPTION_A', "Cats")
#option_b = os.getenv('OPTION_B', "Dogs")
hostname = socket.gethostname()

log = DefaultLogger("app")
gunicorn_error_logger = DefaultLogger("gunicorn.error")
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.INFO)

class InferenceBlob():
    def __init__(self):
        self.model = InferenceModel.from_folder("../llms/s3_outputs/pygmalion-2-7b-sep17")
        self.personas = ChatPersonas.from_folder("../llms/artifacts/personas", "llama2")
        log.info(self.model)
        log.info(self.personas)

inference = InferenceBlob()

@app.route('/predict', methods=['POST'])
def predict():
    input_message = request_body["message"]
    chat_history = request_body["chat_history"]
    persona_id = request_body["persona_id"] if "persona_id" in request_body else None

    persona = inference.personas.get(persona_id)
    
    result = InferenceEngine.predict(input_message, chat_history, inference.model, persona)
    return result


@app.route('/')
def hello():
    input_message = "Tell me a joke!"
    chat_history = ""
    persona_id = "yuki_hinashi_en"

    persona = inference.personas.get(persona_id)
    
    result = InferenceEngine.predict(input_message, chat_history, inference.model, persona)
    return result.reply_message
    #return "Hello World!"


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=False, threaded=True)
