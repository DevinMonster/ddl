import numpy as np
import torchvision.datasets.voc
from torch.utils.data import Dataset, Subset
from torchvision.datasets import VOCSegmentation


class VOCIncrementSegmentation(Dataset):
    '''
    VOC2012 for increment semantic segmentation
        Args:
            path: voc dataset path
            year: year of competition
            is_train: train mode
            download: if you haven't downloaded dataset open it
            img_transform: transform img
            msk_transform: transform mask
            new_labels: labels of classes learn in current stage
            old_labels: labels of 0...current-1 stages
    '''

    def __init__(self, path, year="2012", is_train=True, download=False,
                 img_transform=None, msk_transform=None, new_labels=None,
                 old_labels=None):
        # check args
        assert year in torchvision.datasets.voc.DATASET_YEAR_DICT, \
            f"we don't have dataset in {year}"
        super().__init__()

        voc = VOCSegmentation(path, year, "train" if is_train else "val",
                              download, img_transform, msk_transform)
        self.dataset = voc  # from scratch
        # increment learning
        if new_labels is not None:
            new_labels = self._true_labels(new_labels)
            # filter index of
            idx = filter_images(voc, new_labels, old_labels)
            self.dataset = Subset(voc, idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]  # img, msk

    @staticmethod
    def _true_labels(labels):
        return [0] + [v for v in labels if v != 0] if labels else []


def filter_images(dataset, new_labels, old_labels, overlap=True):
    """
    从对应的dataset中筛出labels对应的
    """
    tot_labels = set(new_labels + old_labels + [0, 255])

    # 当前图像如果没有
    def fil(c):
        f = any((x != 0 and x in new_labels) for x in c)
        if not overlap:
            return f and all(x in tot_labels for x in c)
        return f

    ids = []

    for i, img, mask in enumerate(dataset):
        # all the current img classes in cls
        cls = np.unique(np.array(mask))
        # filter
        if fil(cls):
            ids.append(i)
    return ids


idx2classes = {
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
