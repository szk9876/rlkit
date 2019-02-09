import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.env_replay_buffer import get_dim
from gym.spaces import Box, Discrete, Tuple

class PathReplayBuffer(ReplayBuffer):
    """
    A replay buffer that keeps track of paths.
    max_replay_buffer_size: maximum number of paths
    """
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim, max_path_length
    ):
        self._ob_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._max_path_length = max_path_length
        self._obs = np.zeros((max_replay_buffer_size, max_path_length, observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, max_path_length, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, max_path_length, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, max_path_length, 1))
        self._terminals = np.zeros((max_replay_buffer_size, max_path_length, 1), dtype='uint8')
        self._top = 0
        self._size = 0
        
    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        goals = path["goals"]
        path_len = len(rewards)
        assert path_len == self._max_path_length

        self._obs[self._top] = obs
        self._actions[self._top] = actions
        self._rewards[self._top] = rewards
        self._terminals[self._top] = terminals
        self._next_obs[self._top] = next_obs
        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
            
    def add_sample(self, **kwargs):
        pass
            
    def terminate_episode(self):
        pass
    
    def num_steps_can_sample(self):
        return self._size * self._max_path_length

    def random_batch(self, batch_size):
        path_indices = np.random.randint(low=0, high=self._size, size=batch_size)
        time_indices = np.random.randint(low=0, high=self._max_path_length, size=batch_size)
        indices = np.concatenate([path_indices, time_indices], axis=0)
        return dict(
            observations=self._obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )


class EnvPathReplayBuffer(PathReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, max_path_length, env
    ):
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            max_path_length=max_path_length,
        )
