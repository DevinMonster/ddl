import numpy as np
import torchvision.datasets.voc
from torch.utils.data import Dataset, Subset
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm


class VOCIncrementSegmentation(Dataset):
    '''
    VOC2012 for increment semantic segmentation
        Args:
            path: voc dataset path
            year: year of competition
            is_train: train mode
            download: if you haven't downloaded dataset open it
            transforms: transform both img, and it's mask
            img_transform: transform img
            msk_transform: transform mask
            new_labels: labels of classes learn in current stage
            old_labels: labels of 0...current-1 stages
    '''

    def __init__(self, path, year="2012", is_train=True,
                 download=False, transforms=None, img_transform=None,
                 msk_transform=None, new_labels=None, old_labels=None):
        # check args
        assert year in torchvision.datasets.voc.DATASET_YEAR_DICT, \
            f"we don't have dataset in {year}"
        super().__init__()
        self.transforms = transforms
        self.img_transform = img_transform
        self.msk_transform = msk_transform

        voc = VOCSegmentation(path, year, "train" if is_train else "val", download)

        # filter index of
        if old_labels is None: old_labels = []
        new_labels = self._true_labels(new_labels)
        old_labels = self._true_labels(old_labels)
        # filter index of
        idx = filter_images(voc, new_labels, old_labels)
        self.dataset = Subset(voc, idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, msk = self.transforms(*self.dataset[idx])
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.msk_transform is not None:
            msk = self.msk_transform(msk)
        return img, msk

    @staticmethod
    def _true_labels(labels):
        return [0] + [v for v in labels if v != 0] if labels else []


def filter_images(dataset, new_labels, old_labels=None):
    # Filter images without any label in LABELS (using labels not reordered)
    n_lbs = set(new_labels)
    n_lbs.remove(0)
    tot_lbs = set(new_labels + old_labels + [0, 255])
    idx = []

    def idx_in_labels(cls):
        return any(c in n_lbs for c in cls) and \
            all(c in tot_lbs for c in cls)

    for i, (_, msk) in tqdm(enumerate(dataset)):
        cls = np.unique(np.array(msk))
        if idx_in_labels(cls):
            idx.append(i)
    return idx


voc_idx2classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}
