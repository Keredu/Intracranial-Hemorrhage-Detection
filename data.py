import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
from torchvision import transforms

class IHDataset(Dataset):
    """Intracranial Hemorrhage dataset."""
    def __init__(self, root_dir, stage='train', transform=None):
        self.root_dir = root_dir
        self.stage = stage
        self.transform = transform
        self.data_dir = os.path.join(self.root_dir, self.stage)
        #x = 180000 if stage == 'valid' or stage == 'test' else 180000
        #y = 30 if stage == 'valid' or stage == 'test' else 2
        with open(os.path.join(root_dir, f'annots/{stage}.csv'), 'r') as csv:
            lines = [line.strip().split(',') for line in csv.readlines()]
            header, lines = lines[0], lines[1:]
            self.annots = {k:[] for k in header}
            for line in lines:#[:x:y]:
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


class IHTestDataset(Dataset):
    """Intracranial Hemorrhage dataset."""
    def __init__(self, root_dir, stage='', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_dir = root_dir if root_dir[-1] == '/' else root_dir + '/'
        import glob
        self.data = glob.glob(self.data_dir + '*')
        self.idx_to_img_path = dict(enumerate(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.idx_to_img_path[idx]
        sample = Image.open(img_path)
        if self.transform:
            sample = self.transform(sample)
        return sample, img_path


def get_datasets(conf):
    dataset = conf['data']['name']
    root_dir = conf['data']['path']
    train_transform = conf['train_transform']
    valid_transform = conf['valid_transform']
    test_transform = conf['test_transform']
    if dataset == 'IHDataset':
        train_dataset = IHDataset(root_dir=root_dir,
                                  stage='train',
                                  transform=train_transform)
        valid_dataset = IHDataset(root_dir=root_dir,
                                  stage='valid',
                                  transform=valid_transform)
        test_dataset = IHDataset(root_dir=root_dir,
                                  stage='test',
                                  transform=test_transform)
    elif dataset == 'IHTestDataset':
        return IHTestDataset(root_dir=root_dir,
                             transform=valid_transform)
    else:
        print('Dataset {dataset} not supported.')
        exit()
    return train_dataset, valid_dataset, test_dataset


def get_dataloaders(conf):
    train_dataset = conf['train_dataset']
    valid_dataset = conf['valid_dataset']
    test_dataset = conf['test_dataset']
    batch_size = conf['data']['batch_size']
    num_workers = conf['data']['num_workers']
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size= batch_size,
                                               num_workers=num_workers,
                                               shuffle=False,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size= batch_size,
                                              num_workers=num_workers,
                                              shuffle=False,
                                              pin_memory=True)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    '''
    print(os.listdir('./data/windowed'))
    ds = IHDataset(root_dir='./data/windowed/', stage='valid')
    sample, target = ds[2]
    arr = np.transpose(255 * (0.5 * sample + 0.5), (1,2,0))
    im = Image.fromarray(np.uint8(arr))
    im.save('example.jpg')
    print(target)
    print(len(ds))
    '''
    ds = IHTestDataset(root_dir='../patients_windowed/001_1/')
    sample, img_path = ds[0]
    print('Img path:', img_path)
    arr = sample
    im = Image.fromarray(np.uint8(arr))
    im.save('example.jpg')
