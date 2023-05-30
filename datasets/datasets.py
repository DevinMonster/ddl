import torchvision.datasets.voc
from torch.utils.data import Dataset, Subset
from torchvision.datasets import VOCSegmentation


class VOCIncrementSegmentation(Dataset):
    '''
    VOC2012 for segmentation
    Args:
        path:             voc2012 dataset path

    '''

    def __init__(self, path, year="2012",
                 is_train=True, download=False, transform=None,
                 new_labels=None, old_labels=None, idx_path=None,
                 mask=True, overlap=True):
        # check args
        assert year in torchvision.datasets.voc.DATASET_YEAR_DICT, \
            f"we don't have dataset in {year}"
        super().__init__()

        voc = VOCSegmentation(path, year, "train" if is_train else "val",
                              download=download, transform=transform)
        self.dataset = voc  # from scratch
        # increment learning
        if new_labels is not None:
            if old_labels is None: old_labels = []
            self.new_labels = self._true_labels(new_labels)
            self.old_labels = self._true_labels(old_labels)
            # [0, old, new]
            self.tot_labels = self.old_labels + self.old_labels[1:]
            # TODO:take index
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def _true_labels(labels):
        return [0] + [v for v in labels if v != 0]


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
