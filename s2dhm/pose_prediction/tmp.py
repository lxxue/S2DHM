import torch
from torch import nn
from torch import optim

class Net(nn.Module):
    def __init__(self, n):
        super(Net, self).__init__()
        self.nets = []
        for i in range(n):
            self.nets.append(nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1)))
    
    def forward(self, x):
        y = torch.zeros_like(x)
        for i in range(len(x)):
            y[i] = self.nets[i](x[i])

        return y 

if __name__ == "__main__":
    model = Net(2)
    x = torch.ones(2, 3, 1)
    z = torch.ones(2, 3, 1) * 2
    criterion = nn.MSELoss()
    param = []
    for net in model.nets:
        param += list(net.parameters())
    opt = optim.SGD(param, lr=0.001)
    for i in range(5):
        opt.zero_grad()
        y = model(x)
        loss = criterion(y, z)
        loss.backward()
        opt.step() 
        print(loss.item())


