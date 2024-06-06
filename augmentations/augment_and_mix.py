import random

import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta
from torchvision import transforms

from augmentations import augmentations


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation.
    """
    def __init__(self, dataset, preprocess, k, alpha, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.k = k
        self.alpha = alpha
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return augmentAndMix(x, self.k, self.alpha, self.preprocess), y
        else:
            return (self.preprocess(x), 
                    augmentAndMix(x, self.k, self.alpha, self.preprocess),
                    augmentAndMix(x, self.k, self.alpha, self.preprocess)), y

    def __len__(self):
        return len(self.dataset)

def augmentAndMix(x_orig, k, alpha, preprocess):
    # k : number of chains
    # alpha : sampling constant

    x_temp = x_orig # back up for skip connection

    x_aug = torch.zeros_like(preprocess(x_orig))
    mixing_weight_dist = Dirichlet(torch.empty(k).fill_(alpha))
    mixing_weights = mixing_weight_dist.sample()

    for i in range(k):
        sampled_augs = random.sample(augmentations, k)
        aug_chain_length = random.choice(range(1,k+1))
        aug_chain = sampled_augs[:aug_chain_length]

        for aug in aug_chain:
            severity = random.choice(range(1,6))
            x_temp = aug(x_temp, severity)

        x_aug += mixing_weights[i] * preprocess(x_temp)

    skip_conn_weight_dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    skip_conn_weight = skip_conn_weight_dist.sample()

    x_augmix = skip_conn_weight * x_aug + (1 - skip_conn_weight) * preprocess(x_orig)

    return x_augmix