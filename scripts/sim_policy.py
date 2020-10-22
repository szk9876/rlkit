from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger

import numpy as np
import rlkit.torch.pytorch_util as ptu

import gym

from rlkit.torch.url.envs.disc_wrapper import DiscriminatorWrappedEnv

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']

    # wrapped_env = gym.make('HalfCheetahHolePositions-v{}'.format(1))
    # disc = data['env'].disc
    # reward_params = data['env'].reward_params
    # unsupervised_reward_weight = data['env'].unsupervised_reward_weight
    # reward_weight = data['env'].reward_weight
    # env = DiscriminatorWrappedEnv(wrapped_env=wrapped_env,
    #                              disc=disc,
    #                              reward_params=reward_params,
    #                              unsupervised_reward_weight=unsupervised_reward_weight,
    #                              reward_weight=reward_weight)
    # env = data['env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True, 1)
        policy.cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
        skill = np.random.randint(0, args.num_skills)
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=True,
            skill=skill,
            deterministic=True
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--num_skills', type=int, default=10)
    args = parser.parse_args()
    
    ptu.set_gpu_mode(True, 1)
    simulate_policy(args)
