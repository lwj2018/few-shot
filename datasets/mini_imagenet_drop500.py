import os.path as osp
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

image_root = '/home/liweijie/Data/miniImagenet/images'
csv_root = '/home/liweijie/projects/few-shot/csv'
# !!Only use the last 100 samples which have not been used in training procedure

class MiniImageNet2(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(csv_root, setname + '.csv')
        lines_total = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        lines = []
        tlen = int(len(lines_total)/600)
        print(tlen)
        for i in range(tlen):
            lines.extend(lines_total[i*600+500:(i+1)*600])

        data = []
        label = []

        for l in lines:
            name, lb = l.split(',')
            path = osp.join(image_root, name)
            lb = int(lb)
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform_train = transforms.Compose([
            transforms.Resize(90),
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(84,padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image1 = self.transform_train(Image.open(path).convert('RGB'))
        image2 = self.transform_test(Image.open(path).convert('RGB'))
        image = torch.cat([image1, image2])
        return image, label

