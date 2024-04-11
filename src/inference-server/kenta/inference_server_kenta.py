import argparse
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.outputs import RequestOutput
from vllm.utils import random_uuid

from post_processors import CompleteSentenceProcessor, BestOfProcessor


class ChatCompletionsRequest(BaseModel):
    prompt: str


class AsyncInferenceService:
    def __init__(self):
        self.engine = None
        self._try_load_models()

    def _try_load_models(self):
        engine_args = AsyncEngineArgs(
            model='TheBloke/Llama-2-13B-chat-GPTQ',
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
        self.processors = [
            CompleteSentenceProcessor(),
            BestOfProcessor('ChaiML/reward_models_100_170000000_cp_498032'),
        ]

    @staticmethod
    def get_sampling_params(user_params: dict | None = None):
        params = {
            'use_beam_search': False,
            'temperature': 1.0,
            'top_p': 0.9,
            'n': 4,
            'max_tokens': 128,
        }
        if user_params:
            params.update(user_params)
        return params

    async def completions(self, request: ChatCompletionsRequest, raw_request: Request):
        request_id = random_uuid()
        prompt = request.prompt
        sampling_params = InferenceServerDeployment.get_sampling_params()
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
        processed_outputs = outputs[:]
        for processor in self.processors:
            processed_outputs = processor.process(processed_outputs)

        ret = {
            'messages': [
                {
                    'content': processed_outputs[0],
                },
                {
                    'content': outputs,
                },
            ]
        }
        return JSONResponse(ret)


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--log-level', type=str, default='debug')
    parser.add_argument('--workers', type=int, default=1)
    return parser.parse_args()


app = FastAPI()
server = InferenceServerDeployment()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    #allow_credentials=True,
)

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
