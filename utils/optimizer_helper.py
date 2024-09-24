import torch
from .ConstrainedSGD import ConstrainedSGD


def get_optimizer(parameters, model, config):
    if config.type == "AdamW":
        return torch.optim.AdamW(parameters, **config.kwargs)
    elif config.type == "Adam":
        return torch.optim.Adam(parameters, **config.kwargs)
    elif config.type == "SGD":
        return torch.optim.SGD(parameters, **config.kwargs)
    elif config.type == "ConstrainedSGD":
        return ConstrainedSGD(model.named_parameters(), lr=1)
    else:
        raise NotImplementedError
