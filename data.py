import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
from torchvision import transforms

class IHDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, stage='train', transform=None):
        self.root_dir = root_dir
        self.stage = stage
        self.transform = transform
        self.data_dir = os.path.join(self.root_dir, self.stage)

        with open(os.path.join(root_dir, f'annots/{stage}.csv'), 'r') as csv:
            lines = [line.strip().split(',') for line in csv.readlines()]
            header, lines = lines[0], lines[1:]
            self.annots = {k:[] for k in header}
            for line in lines:
                for k,v in zip(header,line):
                    self.annots[k].append(v)

    def __len__(self):
        return len(self.annots['ID'])

    def __getitem__(self, idx):
        img_name = f'{self.annots["ID"][idx]}.png'
        img_path = os.path.join(self.data_dir, img_name)
        sample = Image.open(img_path)

        if self.transform:
            sample = self.transform(sample)

        target = int(self.annots['IH'][idx])
        return sample, target


def get_datasets(conf):
    dataset = conf['data']['name']
    root_dir = conf['data']['path']
    train_transform = conf['train_transform']
    valid_transform = conf['valid_transform']
    if dataset == 'IHDataset':
        train_dataset = IHDataset(root_dir=root_dir,
                                  stage='train',
                                  transform=train_transform)
        valid_dataset = IHDataset(root_dir=root_dir,
                                  stage='valid',
                                  transform=valid_transform)
    else:
        print('Dataset {dataset} not supported.')
        exit()
    return train_dataset, valid_dataset


def get_dataloaders(conf):
    train_dataset = conf['train_dataset']
    valid_dataset = conf['valid_dataset']
    batch_size = conf['data']['batch_size']
    num_workers = conf['data']['num_workers']
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True ,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size= batch_size,
                                               num_workers=num_workers,
                                               shuffle=False,
                                               pin_memory=True)
    return train_loader, valid_loader


if __name__ == '__main__':
    print(os.listdir('./data/windowed'))
    ds = IHDataset(root_dir='./data/windowed/', stage='valid')
    sample, target = ds[2]
    arr = np.transpose(255 * (0.5 * sample + 0.5), (1,2,0))
    im = Image.fromarray(np.uint8(arr))
    im.save('example.jpg')
    print(target)
    print(len(ds))