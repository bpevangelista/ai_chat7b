import argparse
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.outputs import RequestOutput
from vllm.utils import random_uuid

from personas import ChatPersonas
from post_processors import CompleteSentenceProcessor, BestOfProcessor, SingleResponseProcessor

# MODEL_NAME = 'TheBloke/Llama-2-13B-chat-GPTQ',
MODEL_NAME = 'TheBloke/MythoMax-L2-13B-GPTQ'


# MODEL_NAME = 'TheBloke/Mythalion-13B-GPTQ'

class ChatCompletionsRequest(BaseModel):
    persona_id: str
    prompt: str
    chat_history: list[str]


class ChatCompletionsReply:
    reply: list[str]


class AsyncInferenceService:
    def __init__(self):
        self.engine = None
        self._try_load_models()

    def _try_load_models(self):
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            # model='facebook/opt-125m',
            dtype='float16',
            quantization='gptq',
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def generate(self, request_id: str, prompt: str, sampling_params: SamplingParams):
        generator = self.engine.generate(request_id=request_id, prompt=prompt, sampling_params=sampling_params)
        return generator


class InferenceServerDeployment:
    def __init__(self):
        self.inference = AsyncInferenceService()
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #    model=MODEL_NAME,
        #    trust_remote_code=True,
        # )
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        self.processors = [
            SingleResponseProcessor(),
            CompleteSentenceProcessor(),
            BestOfProcessor('ChaiML/reward_models_100_170000000_cp_498032'),
        ]
        self.personas = ChatPersonas()
        # self.personas.load_all_from_folder('./personas')
        self.personas.load_all_from_folder('./')
        self.personas.load_from_gdoc('custom1', '1tRxEPX-b5nInMVdsr7hkIGrQ4h8Jm-x1-97yZx6dzwQ')
        self.personas.load_from_gdoc('custom2', '18HHK9UrT-OoezSULAJjil8fzWvs8vu_SS2xq5yVyqtA')

    @staticmethod
    def get_sampling_params(user_params: dict | None = None):
        params = {
            'use_beam_search': False,
            'temperature': 0.7,
            'top_p': 0.8,
            'n': 4,
            'max_tokens': 64,

            # Trying
            'presence_penalty': 0.5,
            'frequency_penalty': 0.5,
            'stopping_words': ['<\s>'],
        }
        if user_params:
            params.update(user_params)
        return params

    async def completions(self, request: ChatCompletionsRequest, raw_request: Request):
        if not request.prompt and not request.chat_history:
            prologue, first_message = self.personas.get_first_message(request.persona_id)
            return JSONResponse({'reply': [prologue, first_message]})

        print('\nDEBUG')
        print('----------------------------------------------------------------------')
        print(request.chat_history)
        print('----------------------------------------------------------------------\n')

        request_id = random_uuid()
        # prompt = request.prompt
        prompt = self.personas.build_prompt(request.prompt, request.chat_history, request.persona_id)
        # max_tokens = len(self.tokenizer.tokenize(prompt)) + 32
        # max_tokens = 32
        sampling_params = InferenceServerDeployment.get_sampling_params()
        # sampling_params = SamplingParams(**sampling_params, max_tokens=max_tokens)
        sampling_params = SamplingParams(**sampling_params)

        # Process input

        result_generator = self.inference.generate(request_id, prompt, sampling_params)

        final_output: RequestOutput = None
        async for result in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.inference.engine.abort(request_id)
                return JSONResponse(content='Client disconnected', status_code=HTTPStatus.BAD_REQUEST)
            final_output = result
        assert final_output is not None

        prompt = final_output.prompt

        # TODO I already have it tokenized, no need to tokenize again

        # Process output
        outputs = [output.text for output in final_output.outputs]
        print('DEBUG')
        print(outputs)
        print('')

        processed_outputs = outputs[:]
        for processor in self.processors:
            processed_outputs = processor.process(processed_outputs)

        return JSONResponse({
            'reply': processed_outputs[0],
        })


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--log-level', type=str, default='debug')
    parser.add_argument('--workers', type=int, default=1)
    return parser.parse_args()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_methods=["*"],
    # allow_headers=["*"],
)
server = InferenceServerDeployment()


@app.post(path='/v1/completions', status_code=200)
async def completions(request: ChatCompletionsRequest, raw_request: Request):
    return await server.completions(request, raw_request)


if __name__ == '__main__':
    pargs = handle_args()

    uvicorn.run(
        app,
        host=pargs.host,
        port=pargs.port,
        log_level=pargs.log_level,
        timeout_keep_alive=60,
        loop='uvloop',
    )
