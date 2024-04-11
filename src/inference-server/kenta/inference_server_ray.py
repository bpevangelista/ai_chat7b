import argparse

import kserve
from ray import serve
from starlette.requests import Request


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 8, "num_gpus": 1})
class InferenceServerDeployment:
    def __init__(self):
        pass

    def completions(self, text: str) -> str:
        return 'foo'

    async def __call__(self, request: Request) -> str:
        english_text: str = await request.json()
        return self.completions(english_text)


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--workers', type=int, default=1)
    return parser.parse_args()


"""
if __name__ == '__main__':
    pargs = handle_args()

    deployment = InferenceServerDeployment.bind()
    kserve.ModelServer(
        http_port=pargs.port,
        workers=pargs.workers,
        enable_grpc=False,
    ).start({'/completions': InferenceServerDeployment})
"""

deployment = InferenceServerDeployment.bind()