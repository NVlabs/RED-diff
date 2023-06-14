import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler, Subset
from .lmdb_dataset import LMDBDataset




def get_ffhq_dataset(root, split, transform='default', subset=-1, **kwargs):
    if transform == 'default':
        transform = transforms.ToTensor()
    dset = LMDBDataset(root, 'ffhq', split, transform)
    if isinstance(subset, int) and subset > 0:
        dset = Subset(dset, list(range(subset)))
    else:
        assert isinstance(subset, list)
        dset = Subset(dset, subset)
    return dset


def get_ffhq_loader(dset, *, batch_size, num_workers, shuffle, drop_last, pin_memory, **kwargs):
    sampler = DistributedSampler(dset, shuffle=shuffle, drop_last=drop_last)
    loader = DataLoader(
        dset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, sampler=sampler, pin_memory=pin_memory, persistent_workers=True
    )
    return loader

