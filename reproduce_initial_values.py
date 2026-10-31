import numpy as np
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import Episode, InfiniteBuffer, ReplayBuffer
from d3rlpy.metrics import InitialStateValueEstimationEvaluator


class ValueFunction:
    def predict(self, observations):
        return np.zeros((len(observations), 1), dtype=np.float32)

    def predict_value(self, observations, actions):
        return observations[:, 0]


episodes = []
for length, initial_value in [(2, 2.0), (2049, 6.0)]:
    observations = np.full((length, 1), 100.0, dtype=np.float32)
    observations[0] = initial_value
    episodes.append(Episode(
        observations=observations,
        actions=np.zeros((length, 1), dtype=np.float32),
        rewards=np.zeros((length, 1), dtype=np.float32),
        terminated=True,
    ))
dataset = ReplayBuffer(
    InfiniteBuffer(), episodes=episodes,
    action_space=ActionSpace.CONTINUOUS, action_size=1,
)
print(InitialStateValueEstimationEvaluator()(ValueFunction(), dataset))
# Expected: (2 + 6) / 2 = 4.0
# Current master: (2 + 6 + 100 + 100) / 4 = 52.0
