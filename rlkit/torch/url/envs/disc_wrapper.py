from multiworld.core.multitask_env import MultitaskEnv
from rlkit.envs.wrappers import ProxyEnv
from rlkit.util.io import load_local_or_remote_file
import numpy as np
from rlkit.torch.torch_rl_algorithm import np_to_pytorch_batch
from rlkit.torch import pytorch_util as ptu
import copy
from gym.spaces import Discrete, Dict

class DiscriminatorWrappedEnv(ProxyEnv, MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
            disc,
            mode='train',
            reward_params=None,
    ):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        if type(disc) is str:
            self.disc = load_local_or_remote_file(disc)
        else:
            self.disc = disc
        self._num_skills = self.disc.num_skills
        self._p_z = np.full(self._num_skills, 1.0 / self._num_skills)
        self.task = {'context': -1}
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get('type', 'diayn')

        spaces = copy.deepcopy(self.wrapped_env.observation_space.spaces)
        spaces['context'] = Discrete(n=self._num_skills)
        self.observation_space = Dict(spaces)

        assert self.reward_type == 'wrapped_env' or self.reward_type == 'diayn'

    def reset(self):
        obs = self.wrapped_env.reset()
        z = self._sample_z(1)
        task = {}
        task['context'] = z
        self.task = task
        return self._update_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        reward = self.compute_reward(action,
                                     {'observation': new_obs['observation'],
                                      'context': new_obs['context']})
        return new_obs, reward, done, info

    def _sample_z(self, batch_size):
        """Samples z from p(z), using probabilities in self._p_z."""
        return np.random.choice(self._num_skills, size=batch_size, replace=True, p=self._p_z)

    def _update_obs(self, obs):
        obs = {**obs, **self.task}
        return obs

    def get_goal(self):
        raise NotImplementedError()

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs):
        if self.reward_type == 'diayn':
            skill = obs['context']
            next_ob = obs['observation']

            cross_entropy = self.disc.evaluate_cross_entropy(inputs=next_ob, labels=skill)
            p_z = self._p_z[skill.astype(np.int)]
            log_p_z = np.log(p_z)

            assert cross_entropy.shape == log_p_z.shape
            reward = -1 * cross_entropy - log_p_z
            return reward
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError()

    def sample_goals(self, batch_size):
        return {
            'context': self._sample_z(batch_size)
        }

    def sample_goal(self):
        goals = self.sample_goals(1)
        return self.unbatchify_dict(goals, 0)

    def fit(self, replay_buffer):
        return self.disc.fit(replay_buffer)

    @property
    def context_dim(self):
        return self._num_skills


