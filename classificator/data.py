import sys
sys.path.append(".")

import classificator
import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,
                                        download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=classificator.batch_size,shuffle=True,num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=classificator.batch_size,shuffle=False,num_workers=2)