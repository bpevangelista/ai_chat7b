from flask import Flask, request
#from flask_cors import CORS

import json, logging, socket

from inference import InferenceEngine, InferenceModel, ChatPersonas, DefaultLogger

app = Flask(__name__)
#cors = CORS(app)

#option_a = os.getenv('OPTION_A', "Cats")
#option_b = os.getenv('OPTION_B', "Dogs")
hostname = socket.gethostname()

log = DefaultLogger("app")
gunicorn_error_logger = DefaultLogger("gunicorn.error")
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.INFO)

class InferenceBlob():
    def __init__(self):
        """
        #self.model = InferenceModel.from_folder("../../llms/s3_outputs/pygmalion-2-7b-sep17")
        self.model = InferenceModel.from_folder("../../llms/s3_outputs/pygmalion-6b-sep01")
        self.personas = ChatPersonas.from_folder("../../llms/artifacts/personas", self.model.model_type)
        """
        self.model = InferenceModel.from_folder("artifacts", False)
        self.personas = ChatPersonas.from_folder("artifacts/personas", self.model.model_type)

        log.info(self.model)
        log.info(self.personas)

inference = InferenceBlob()


@app.route('/predict', methods=['POST'])
def predict():
    log.info(request.json)
    request_body = request.json
    message = request_body["message"]
    chat_history = request_body["chat_history"]
    persona_id = request_body["persona_id"] if "persona_id" in request_body else None

    persona = inference.personas.get(persona_id)

    result = InferenceEngine.predict(message, chat_history, inference.model, persona)
    return json.dumps(result.__json__())


@app.route('/')
def hello():
    message = "Tell me a joke!"
    chat_history = ""
    persona_id = "yuki_hinashi_en"

    persona = inference.personas.get(persona_id)

    result = InferenceEngine.predict(message, chat_history, inference.model, persona)
    return json.dumps(result.__json__())


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=False, threaded=True)
