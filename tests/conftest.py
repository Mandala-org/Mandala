import random
import numpy as np
import torch


def pytest_configure(config):
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_printoptions(precision=5)
