import os
import torch.distributed as dist
from torch.utils.data import Dataset

#SELENE
from datasets.ffhq import get_ffhq_dataset, get_ffhq_loader
from datasets.imagenet import get_imagenet_dataset, get_imagenet_loader
from utils.distributed import get_logger

#LOCAL MACHINE
# from pgdm.datasets.ffhq import get_ffhq_dataset, get_ffhq_loader
# from pgdm.datasets.imagenet import get_imagenet_dataset, get_imagenet_loader
# from pgdm.utils.distributed import get_logger



class ZipDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        assert all(len(dataset) == len(datasets[0]) for dataset in datasets)

    def __getitem__(self, index):
        return [dataset[index] for dataset in self.datasets]

    def __len__(self):
        return len(self.datasets[0])


def build_one_dataset(cfg, dataset_attr='dataset'):
    logger = get_logger('dataset', cfg)
    exp_root = cfg.exp.root
    cfg_dataset = getattr(cfg, dataset_attr)
    try:
        samples_root = cfg.exp.samples_root
        exp_name = cfg.exp.name
        samples_root = os.path.join(exp_root, samples_root, exp_name)
    except Exception:
        samples_root = ''
        logger.info('Does not attempt to prune existing samples (overwrite=False).')
    if "ImageNet" in cfg_dataset.name:
        overwrite = getattr(cfg.exp, 'overwrite', True)
        dset = get_imagenet_dataset(overwrite=overwrite, samples_root=samples_root, **cfg_dataset)
        dist.barrier()
    if "FFHQ" in cfg_dataset.name:
        dset = get_ffhq_dataset(**cfg_dataset)

    return dset


def build_loader(cfg, dataset_attr='dataset'):
        
    if type(dataset_attr) == list:
        dsets = []
        for da in dataset_attr:
            cfg_dataset = getattr(cfg, da)
            dset = build_one_dataset(cfg, dataset_attr=da)
            dsets.append(dset)
        dsets = ZipDataset(dsets)
        if "ImageNet" in cfg_dataset.name:
            loader = get_imagenet_loader(dsets, **cfg.loader)
        elif "FFHQ" in cfg_dataset.name:
            loader = get_ffhq_loader(dsets, **cfg.loader)
    else:
        cfg_dataset = getattr(cfg, dataset_attr)
        dset = build_one_dataset(cfg, dataset_attr=dataset_attr)
        if "ImageNet" in cfg_dataset.name:
            loader = get_imagenet_loader(dset, **cfg.loader)
        elif "FFHQ" in cfg_dataset.name:
            loader = get_ffhq_loader(dset, **cfg.loader)
            

    return loader
