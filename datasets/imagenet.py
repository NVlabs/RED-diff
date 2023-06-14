import os
from functools import partial
from typing import Any, Tuple

import torch
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
import PIL
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets.imagenet import (
    META_FILE,
    check_integrity,
    load_meta_file,
    parse_devkit_archive,
    parse_train_archive,
    parse_val_archive,
    verify_str_arg,
)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".JPEG")


def center_crop_arr(pil_image, image_size=256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root: str, split: str = "train", subset_txt='', samples_root="", meta_root="", **kwargs):
        if split == "train" or split == "val":
            root = os.path.join(root, "imagenet")
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val", "custom"))

        self.parse_archives()
        
        try:
            wnid_to_classes = load_meta_file(self.root)[0]
        except Exception:
            wnid_to_classes = load_meta_file(meta_root)[0]
            
        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

        if len(subset_txt) > 0:
            with open(subset_txt, "r") as f:
                lines = f.readlines()
            self.samples = []
            for line in lines:
                idx = line.split()[0]
                if self.split == "custom":
                    idx = idx[:-5] + '.png'
                path = os.path.join(self.split_folder, idx)
                label = int(line.split()[1])
                self.samples.append((path, label))

            self.targets = [s[1] for s in self.samples]

        if len(samples_root) > 0:
            wnid_exists = [entry.name for entry in os.scandir(samples_root) if entry.is_dir()]
            
            #check fot empty dirs
            wnid_exists = [x for x in wnid_exists if len(os.listdir(os.path.join(samples_root,x)))>0]
            #import pdb; pdb.set_trace()
            
            wnid_to_idx = {wnid: self.wnid_to_idx[wnid] for wnid in wnid_exists}

            samples_done = self.make_dataset(samples_root, wnid_to_idx, extensions=IMG_EXTENSIONS)
            samples_done = [s[0].split("/")[-1].split(".")[-2] for s in samples_done]
            samples = []
            for sample in self.samples:
                k = [s in sample[0] for s in samples_done]
                if not any(k):
                    samples.append(sample)
            self.samples = samples
            self.targets = [s[1] for s in self.samples]

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            try:
                parse_devkit_archive(self.root)
            except Exception:
                pass

        if not os.path.isdir(self.split_folder):
            if self.split == "train":
                parse_train_archive(self.root)
            elif self.split == "val":
                parse_val_archive(self.root)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        class_wnid = path.split("/")[-2]
        name = path.split("/")[-1].split(".")[0]

        return sample, target, {"class_id": class_wnid, "name": name, "index": index}

    @property
    def split_folder(self) -> str:
        if self.split == "train" or self.split == "val":
            return os.path.join(self.root, self.split)
        else:
            return self.root

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


def get_imagenet_dataset(
    *, root, split, image_size, subset_txt=None, overwrite=False, samples_root="", meta_root="", transform='', **kwargs
):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transform == 'diffusion':
        transform = transforms.Compose([partial(center_crop_arr, image_size=image_size), transforms.ToTensor()])
    elif transform == 'ca_imagenet':
        transform = transforms.Compose([
            transforms.Resize(256, PIL.Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif transform == 'ca_cropped':
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    elif transform == 'identity':
        transform = transforms.Compose([transforms.PILToTensor()])
    elif transform == 'isc_cropped':
        transform = transforms.Compose([partial(center_crop_arr, image_size=image_size), transforms.ToTensor(), transforms.ConvertImageDtype(torch.uint8)])
    else:
        raise ValueError(f'Transform {transform} does not exist.')
    dset = ImageNet(
        root=root,
        split=split,
        transform=transform,
        subset_txt=subset_txt,
        samples_root=samples_root if not overwrite and len(os.listdir(samples_root))>0 else "",
        meta_root=meta_root,
    )
    return dset


def get_imagenet_loader(dset, *, batch_size, num_workers, shuffle, drop_last, pin_memory, **kwargs):
    sampler = DistributedSampler(dset, shuffle=shuffle, drop_last=drop_last)
    loader = DataLoader(
        dset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, sampler=sampler, pin_memory=pin_memory, persistent_workers=True
    )
    return loader
