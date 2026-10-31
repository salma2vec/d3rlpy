"""Offline public-API reproductions; this file is excluded from the PR."""
import dataclasses
import json

import numpy as np
import torch

from d3rlpy.algos import DQNConfig
from d3rlpy.dataset import Episode, InfiniteBuffer, ReplayBuffer
from d3rlpy.logging import NoopAdapterFactory
from d3rlpy.metrics import AverageValueEstimationEvaluator
from d3rlpy.models import EncoderFactory, register_encoder_factory
from d3rlpy.models.torch import Encoder


def episode(length, terminated=True, multimodal=False):
    observations = np.zeros((length, 2), dtype=np.float32)
    if multimodal:
        observations = [observations, np.ones((length, 3), dtype=np.float32)]
    return Episode(
        observations=observations,
        actions=np.arange(length, dtype=np.int64).reshape(-1, 1) % 2,
        rewards=np.ones((length, 1), dtype=np.float32),
        terminated=terminated,
    )


results = {}
for length in (1024, 1025, 1026, 2048, 2049):
    for terminated in (False, True):
        data = ReplayBuffer(InfiniteBuffer(), episodes=[episode(length, terminated)])
        algo = DQNConfig().create()
        algo.build_with_dataset(data)
        key = f"length={length},terminated={terminated}"
        try:
            value = AverageValueEstimationEvaluator()(algo, data)
            results[key] = {"result": "ok", "finite": bool(np.isfinite(value))}
        except Exception as error:
            results[key] = {"result": type(error).__name__, "error": str(error)}
print("BOUNDARY_RESULTS=" + json.dumps(results, sort_keys=True))
assert results["length=1025,terminated=False"]["result"] == "IndexError"
assert results["length=2049,terminated=False"]["result"] == "IndexError"
assert all(
    result["result"] == "ok"
    for key, result in results.items()
    if key not in ("length=1025,terminated=False", "length=2049,terminated=False")
)


class ConcatEncoder(Encoder):
    def __init__(self, observation_shape):
        super().__init__()
        size = (
            int(np.prod(observation_shape))
            if isinstance(observation_shape[0], int)
            else sum(int(np.prod(shape)) for shape in observation_shape)
        )
        self.linear = torch.nn.Linear(size, 8)

    def forward(self, observations):
        if not isinstance(observations, torch.Tensor):
            observations = torch.cat(
                [component.flatten(start_dim=1) for component in observations], dim=1
            )
        return self.linear(observations)


@dataclasses.dataclass
class ConcatEncoderFactory(EncoderFactory):
    def create(self, observation_shape):
        return ConcatEncoder(observation_shape)

    @staticmethod
    def get_type():
        return "public_repro_concat"


register_encoder_factory(ConcatEncoderFactory)
data = ReplayBuffer(InfiniteBuffer(), episodes=[episode(4, multimodal=True)])
observations = data.episodes[0].observations
batch = [component[:1] for component in observations]
config = DQNConfig(encoder_factory=ConcatEncoderFactory())
built = config.create()
built.build_with_dataset(data)
try:
    built.predict(batch)
except RuntimeError as error:
    print("BUILD_WITH_DATASET_ERROR=" + str(error))
else:
    raise AssertionError("Expected shape mismatch on upstream")

trained = config.create()
trained.fit(
    data,
    n_steps=1,
    n_steps_per_epoch=1,
    show_progress=False,
    logger_adapter=NoopAdapterFactory(),
)
assert trained.predict(batch).shape == (1,)
print("FIT_WITH_SAME_DATASET_AND_ENCODER=passed")
