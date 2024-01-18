import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()

        # 将预测值和真实值展平
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)

        # 计算交集
        intersection = (y_pred * y_true).sum()

        # 计算 Dice 系数
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )

        # 返回 Dice Loss
        return 1. - dsc
