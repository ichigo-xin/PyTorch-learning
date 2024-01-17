import torch
import torch.nn as nn
import numpy as np
input_feat = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
print(input_feat)
print(input_feat.shape)

kernels = torch.tensor([[[[1, 0], [1, 1]]]], dtype=torch.float32)
print(kernels)
print(kernels.shape)


convTrans = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=0, bias = False)
convTrans.weight=nn.Parameter(kernels)
out_feat = convTrans(input_feat)
print(out_feat)
