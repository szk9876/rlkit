from rlkit.torch.networks import Mlp
from torch.optim import Adam
import torch.nn as nn
from rlkit.torch.torch_rl_algorithm import np_to_pytorch_batch
from rlkit.util.meter import AverageMeter
from rlkit.torch import pytorch_util as ptu
import torch
from tqdm import tqdm


class Discriminator(Mlp):
    def __init__(
            self,
            *args,
            batch_size=256,
            num_batches_per_fit=50,
            num_skills=20,
            sampling_strategy='random',
            sampling_window=10,
            lr=3e-4,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.num_skills = num_skills
        self.batch_size = batch_size
        self.num_batches_per_fit = num_batches_per_fit
        self.sampling_strategy = sampling_strategy
        self.sampling_window = sampling_window

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_function_mean = nn.CrossEntropyLoss(reduction='elementwise_mean')
        self.loss_function = nn.CrossEntropyLoss(reduction='none')
        self.loss_meter = AverageMeter()

    def fit(self, replay_buffer):
        self.train()
        self.loss_meter.reset()

        # t = tqdm(range(self.num_batches_per_fit))
        t = range(self.num_batches_per_fit)
        for i in t:
            if self.sampling_strategy == 'random':
                batch = replay_buffer.random_batch(self.batch_size)
            elif self.sampling_strategy == 'recent':
                batch = replay_buffer.recent_batch(self.batch_size, self.sampling_window)
            else:
                raise ValueError
            batch = np_to_pytorch_batch(batch)
            inputs = batch['observations']
            labels = batch['context'].long()

            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.loss_function_mean(outputs, labels.squeeze(1))
            loss.backward()
            self.optimizer.step()

            self.loss_meter.update(val=loss.item(), n=self.batch_size)

            # print(self.loss_meter.avg)

        self.eval()
        return self.loss_meter.avg

    def evaluate_cross_entropy(self, inputs, labels):
        with torch.no_grad():
            inputs = ptu.from_numpy(inputs)
            labels = ptu.from_numpy(labels).long()
            logits = self.forward(inputs)
            return ptu.get_numpy(self.loss_function(logits, labels.squeeze(1)).unsqueeze(1))
