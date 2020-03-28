import os.path as osp
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

image_root = '/home/liweijie/Data/miniImagenet/images'
csv_root = '/home/liweijie/projects/few-shot/csv'

class MiniImageNet(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(csv_root, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(image_root, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform_train = transforms.Compose([
            transforms.Resize(90),
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
        try:
            image = Image.open(path).convert('RGB')
        except:
            print(path)
        image1 = self.transform_train(image)
        image2 = self.transform_test(image)
        image = torch.cat([image1, image2])
        return image, label

# Test
if __name__ == '__main__':
    dataset = MiniImageNet('trainvaltest')
    # Check every file in the dataset
    for i in range(len(dataset)):
        dataset[i]