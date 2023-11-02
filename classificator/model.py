# import sys
# sys.path.append(".")

# import classificator


import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from torchmetrics.functional import accuracy


class L_model(L.LightningModule):
    def __init__(self, channels, width, height, num_classes, hidden_size=64,lr=2e-4):
        super().__init__()

        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.lr = lr

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels*width*height,hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )
    def forward(self,x):
        x = self.model(x)
        return F.log_softmax(x,dim=1)
    def training_step(self,batch):
        x,y = batch
        logits = self(x)
        loss = F.nll_loss(logits,y)
        return loss
    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = F.nll_loss(logits,y)
        preds = torch.argmax(logits,dim=1)
        acc = accuracy(preds,y, task='multiclass',num_classes=10)
        self.log('val_loss',loss, prog_bar=True)
        self.log('val_acc', acc,prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer



# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3,6,5)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv2 = nn.Conv2d(6,16,5)
#         self.fc1 = nn.Linear(16*5*5,120)
#         self.fc2 = nn.Linear(120,84)
#         self.fc3 = nn.Linear(84,10)

#     def forward(self,x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x,1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x