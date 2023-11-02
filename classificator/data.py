import sys
sys.path.append(".")

import classificator
import torch
import torchvision
import torchvision.transforms as transforms

import os
import lightning as L
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split


from torchvision.datasets import CIFAR10

PATH_DATASETS = os.environ.get('PATH_DATASETS',".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64



class L_data_module(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        )
        self.dims = classificator.dims
        self.num_classes = classificator.num_classes

    def prepare_data(self):
        CIFAR10(self.data_dir,train=True, download=True)
        CIFAR10(self.data_dir,train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=BATCH_SIZE)
    
    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=BATCH_SIZE)
    



# transform = transforms.Compose(
#     [transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
# )

# trainset = torchvision.datasets.CIFAR10(root='./data',train=True,
#                                         download=True,transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset,batch_size=classificator.batch_size,shuffle=True,num_workers=2)
# testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
# testloader = torch.utils.data.DataLoader(testset,batch_size=classificator.batch_size,shuffle=False,num_workers=2)

'''
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classificator.classes[labels[j]]:5s}' for j in range(classificator.batch_size)))
'''
