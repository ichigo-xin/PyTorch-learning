import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        # conv1输出的特征图为222x222大小
        self.fc = nn.Linear(16 * 222 * 222, 10)

    def forward(self, input):
        x = self.conv1(input)
        # 进去全连接层之前，先将特征图铺平
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

transform = transforms.Compose([
        # transforms.RandomResizedCrop((224, 224)),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_my_cnn():


    cifar10_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                   train=False,
                                                   transform=transform,
                                                   target_transform=None,
                                                   download=True)

    dataloader = DataLoader(dataset=cifar10_dataset,  # 传入的数据集, 必须参数
                            batch_size=32,  # 输出的batch大小
                            shuffle=True,  # 数据是否打乱
                            num_workers=0)  # 进程数, 0表示只有主进程
    model = MyCNN()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

    for epoch in range(3):
        for item in dataloader:
            output = model(item[0])
            target = item[1]
            # 使用交叉熵损失函数
            loss = nn.CrossEntropyLoss()(output, target)
            print('Epoch {}, Loss {}'.format(epoch + 1, loss))
            # 以下代码的含义，我们在之前的文章中已经介绍过了
            model.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model, './my_model.pth')


if __name__ == '__main__':
    model_predict = torch.load('./my_model.pth')
    model_predict.eval()
    im1 = Image.open('dog.jpg')
    im2 = Image.open('cat1.jpg')
    im3 = Image.open('plane1.jpg')
    print(model_predict(transform(im1).unsqueeze(0)).argmax())
    print(model_predict(transform(im2).unsqueeze(0)).argmax())
    print(model_predict(transform(im3).unsqueeze(0)).argmax())
