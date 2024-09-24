import logging

from datasets.cifar_dataset import build_cifar10_dataloader
from datasets.custom_dataset import build_custom_dataloader

logger = logging.getLogger("global")


def build(cfg, training, testing, distributed):
    if cfg.get("train", None) and training:
        cfg.update(cfg.get("train", {}))
    elif cfg.get("Val", None) and not training and not testing:
        cfg.update(cfg.get("Val", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training, distributed)
    elif dataset == "cifar10":
        data_loader = build_cifar10_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, distributed=True):
    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(cfg_dataset, training=True, testing=False,
                             distributed=distributed)

    val_loader = None
    if cfg_dataset.get("Val", None):
        val_loader = build(cfg_dataset, training=False, testing=False,
                           distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, training=False, testing=True,
                            distributed=distributed)

    logger.info("build dataset done")
    return train_loader, val_loader, test_loader