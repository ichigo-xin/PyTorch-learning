import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = torch.ones((3, 3))
print(data.device)
# Get: cpu

# 获得device
device = torch.device("cuda:0")

# 将data推到gpu上
data_gpu = data.to(device)
print(data_gpu.device)
# Get: cuda:0
