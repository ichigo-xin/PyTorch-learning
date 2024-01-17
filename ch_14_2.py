import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

alexnet = models.alexnet()
alexnet.load_state_dict(torch.load('alexnet-owt-7be5be79.pth'))

im = Image.open('dog.jpg')

# 提取分类层的输入参数
fc_in_features = alexnet.classifier[6].in_features

# 修改预训练模型的输出分类数
alexnet.classifier[6] = torch.nn.Linear(fc_in_features, 10)
print(alexnet)

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = transform(im).unsqueeze(0)
print(alexnet(input_tensor).argmax())
print(alexnet)

print('-----------------')

transform = transforms.Compose([
    # transforms.RandomResizedCrop((224, 224)),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data',
                                               train=False,
                                               transform=transform,
                                               target_transform=None,
                                               download=True)
dataloader = DataLoader(dataset=cifar10_dataset,  # 传入的数据集, 必须参数
                        batch_size=32,  # 输出的batch大小
                        shuffle=True,  # 数据是否打乱
                        num_workers=0)  # 进程数, 0表示只有主进程

optimizer = torch.optim.SGD(alexnet.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

# 训练3个Epoch
for epoch in range(3):
    for item in dataloader:
        output = alexnet(item[0])
        target = item[1]
        # 使用交叉熵损失函数
        loss = nn.CrossEntropyLoss()(output, target)
        print('Epoch {}, Loss {}'.format(epoch + 1, loss))
        # 以下代码的含义，我们在之前的文章中已经介绍过了
        alexnet.zero_grad()
        loss.backward()
        optimizer.step()

im = Image.open('dog.jpg')
input_tensor = transform(im).unsqueeze(0)
print(alexnet(input_tensor).argmax())

# cifar10_dataset = torchvision.datasets.CIFAR10(root='./data',
#                                                train=False,
#                                                transform=transforms.ToTensor(),
#                                                target_transform=None,
#                                                download=True)
# # 取32张图片的tensor
# tensor_dataloader = DataLoader(dataset=cifar10_dataset,
#                                batch_size=32)
# data_iter = iter(tensor_dataloader)
# img_tensor, label_tensor = next(data_iter)
# print(img_tensor.shape)
# grid_tensor = torchvision.utils.make_grid(img_tensor, nrow=16, padding=2)
# grid_img = transforms.ToPILImage()(grid_tensor)
# # grid_img.show()
