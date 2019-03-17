import numpy as np
from gym.spaces import Dict

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.obs_dict_replay_buffer import flatten_n, flatten_dict
from gym.spaces import Box, Discrete, Tuple


class ObsDictPathReplayBuffer(ReplayBuffer):
    """
    Replay buffer for
        - keeping track of information about paths/trajectories
        - environments where observations are dictionaries
    max_replay_buffer_size: maximum number of paths
    """

    def __init__(
            self,
            max_size,
            max_path_length,
            env,
            observation_key='observation',
            context_key='context',
    ):
        assert isinstance(env.observation_space, Dict)
        self.max_size = max_size
        self.max_path_length = max_path_length
        self.env = env
        self.ob_keys_to_save = [
            observation_key,
            context_key
        ]
        self.observation_key = observation_key
        self.context_key = context_key

        self._action_dim = env.action_space.low.size

        self._actions = np.zeros((max_size, max_path_length, self._action_dim))
        self._terminals = np.zeros((max_size, max_path_length, 1), dtype='uint8')
        self._obs = {}
        self._next_obs = {}
        self.ob_spaces = self.env.observation_space.spaces
        for key in self.ob_keys_to_save:
            assert key in self.ob_spaces, "Key not found in the observation space: {}".format(key)
            dtype = np.float64
            if type(self.ob_spaces[key]) is Box:
                dsize = self.ob_spaces[key].low.size
            elif type(self.ob_spaces[key]) is Discrete:
                dsize = 1
            else:
                raise NotImplementedError

            self._obs[key] = np.zeros((max_size, max_path_length, dsize), dtype=dtype)
            self._next_obs[key] = np.zeros((max_size, max_path_length, dsize), dtype=dtype)

        self._top = 0
        self._size = 0

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(terminals)
        assert path_len == self.max_path_length

        actions = flatten_n(actions)
        obs = flatten_dict(obs, self.ob_keys_to_save)
        next_obs = flatten_dict(next_obs, self.ob_keys_to_save)

        self._actions[self._top] = actions
        self._terminals[self._top] = terminals
        for key in self.ob_keys_to_save:
            self._obs[key][self._top] = obs[key]
            self._next_obs[key][self._top] = next_obs[key]

        self._actions[self._top] = actions

        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self.max_size
        if self._size < self.max_size:
            self._size += 1

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size * self.max_path_length

    def random_batch(self, batch_size):
        path_indices = np.random.randint(low=0, high=self._size, size=batch_size)
        time_indices = np.random.randint(low=0, high=self.max_path_length, size=batch_size)
        return self._get_batch(path_indices, time_indices)

    def recent_batch(self, batch_size, window):
        if self._size < window:
            path_index_low = 0
            path_index_high = self._size
        else:
            path_index_low = self._top - window
            path_index_high = self._top
        path_indices = np.random.randint(low=path_index_low, high=path_index_high, size=batch_size)
        time_indices = np.random.randint(low=0, high=self.max_path_length, size=batch_size)
        return self._get_batch(path_indices, time_indices)

    def _get_batch(self, path_indices, time_indices):
        obs_dict = self._batch_obs_dict(path_indices, time_indices)
        next_obs_dict = self._batch_next_obs_dict(path_indices, time_indices)
        actions = self._actions[path_indices, time_indices]
        rewards = self.env.compute_rewards(actions, next_obs_dict)

        batch = {
            'observations': obs_dict[self.observation_key],
            'actions': actions,
            'rewards': rewards,
            'terminals': self._terminals[path_indices, time_indices],
            'next_observations': next_obs_dict[self.observation_key],
            'context': obs_dict[self.context_key]
        }

        return batch

    def _batch_obs_dict(self, path_indices, time_indices):
        return {
            key: self._obs[key][path_indices, time_indices]
            for key in self.ob_keys_to_save
        }

    def _batch_next_obs_dict(self, path_indices, time_indices):
        return {
            key: self._next_obs[key][path_indices, time_indices]
            for key in self.ob_keys_to_save
        }

    def get_trajectories(self):
        """Returns 3D data: (num_paths, episode_length, num_features)."""
        indices = np.s_[self._top-self._size : self._top]
        obs_dict = {key: self._obs[key][indices] for key in self.ob_keys_to_save}
        actions = self._actions[indices]
        return {'observations': obs_dict[self.observation_key],
                'actions': actions}

    def resample_tasks(self, env):
        num_tasks = self._size
        task_dict = env.sample_goals(num_tasks)
        context = task_dict['context']
        context = np.broadcast_to(context, [num_tasks, self.max_path_length, env.context_dim])
        indices = np.s_[self._top - self._size: self._top]
        self._obs[self.context_key][indices] = context

