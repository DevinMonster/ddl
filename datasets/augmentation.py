import math
import random
import warnings
from typing import List, Callable

import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

'''
由于torchvision自带的transforms不允许同时多个输入
因此我们重写这些类，以便同时处理(img, mask)
'''


class Compose:
    def __init__(self, transforms: List[Callable]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, img, msk=None):
        for t in self.transforms:
            img, msk = t(img, msk)
        if msk is not None: return img, msk
        return img


class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = [size, size]
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img, msk=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if msk is not None:
            return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), \
                F.resized_crop(msk, i, j, h, w, self.size, InterpolationMode.NEAREST)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, msk=None):
        if random.random() < self.p:
            img = F.hflip(img)
            if msk: msk = F.hflip(msk)
        if msk is not None:
            return img, msk
        return img


class ToTensor:
    def __call__(self, img, msk=None):
        img = F.to_tensor(img)
        if msk is not None:
            return img, F.to_tensor(msk)
        return img


class Normalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img, msk=None):
        img = F.normalize(img, self.mean, self.std)
        if msk is not None: return img, msk
        return img
