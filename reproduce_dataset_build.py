"""Run the repository's tuple-observation encoder through both public build paths."""
import argparse

import numpy as np

from d3rlpy.algos import SACConfig
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import Episode, InfiniteBuffer, ReplayBuffer
from d3rlpy.logging import NoopAdapterFactory
from examples.tuple_observation import TupleEncoderFactory

parser = argparse.ArgumentParser()
parser.add_argument("--expect-failure", action="store_true")
args = parser.parse_args()
episode = Episode(
    observations=[np.ones((4, 2), dtype=np.float32), np.ones((4, 3), dtype=np.float32)],
    actions=np.zeros((4, 1), dtype=np.float32),
    rewards=np.ones((4, 1), dtype=np.float32),
    terminated=True,
)
data = ReplayBuffer(
    InfiniteBuffer(), episodes=[episode], action_space=ActionSpace.CONTINUOUS, action_size=1
)
config = SACConfig(
    actor_encoder_factory=TupleEncoderFactory(),
    critic_encoder_factory=TupleEncoderFactory(),
)
built = config.create()
failed = False
try:
    built.build_with_dataset(data)
except ValueError as error:
    failed = True
    print(f"build_with_dataset: {type(error).__name__}: {error}")
assert failed == args.expect_failure
batch = [part[:1] for part in episode.observations]
if not failed:
    assert built.predict(batch).shape == (1, 1)
    print("build_with_dataset: prediction succeeded")
trained = config.create()
trained.fit(
    data, n_steps=1, n_steps_per_epoch=1,
    show_progress=False, logger_adapter=NoopAdapterFactory()
)
assert trained.predict(batch).shape == (1, 1)
print("fit with the same dataset and example encoder: prediction succeeded")
