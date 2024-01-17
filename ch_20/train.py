import torch

from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from dataset import CatSegmentationDataset as Dataset

def data_loaders(args):
    dataset_train = Dataset(
        images_dir=args.images,
        image_size=args.image_size,
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    return loader_train

# args是传入的参数
def main(args):
    loader_train = data_loaders(args)