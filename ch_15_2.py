import random
import numpy as np
import torch
from torch import nn
# Tensorboard
from torch.utils.tensorboard import SummaryWriter


# 模型定义
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, input):
        return (input * self.weight) + self.bias


# 数据
w = 2
b = 3
xlim = [-10, 10]
x_train = np.random.randint(low=xlim[0], high=xlim[1], size=30)
y_train = [w * x + b + random.randint(0, 2) for x in x_train]

# 训练
model = LinearModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)
y_train = torch.tensor(y_train, dtype=torch.float32)

writer = SummaryWriter()

for n_iter in range(500):
    input = torch.from_numpy(x_train)
    output = model(input)
    loss = nn.MSELoss()(output, y_train)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar('Loss/train', loss, n_iter)
