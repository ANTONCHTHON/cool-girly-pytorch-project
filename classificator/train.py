import data as d
import model

import torch.nn as nn
import torch.optim as optim

net = model.Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(d.trainloader,0):
        inputs, labels = data
        
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch +  1 },{i+ 1:5d}] loss:{running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished training')

