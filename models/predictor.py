import torch
from torch import nn
from torch.nn import functional as F


class MlpPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_type, spatial_border):
        """
        Args:
            input_size (int): number of input feature dimension.
            hidden_size (int): number of hidden feature dimension.
            output_size (int): number of output feature dimension.
            pred_type (str): type of prediction, 'regression' or 'classification'.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.pred_type = pred_type
        self.spatial_border = nn.Parameter(torch.tensor(spatial_border), requires_grad=False)

    def forward(self, traj_h):
        pred = self.net(traj_h)
        return pred

    # def loss(self, traj_h, label, denormalize=False, tte_loss=False):
    #     #针对tte任务设置的loss，由于 pred 有两个输出维度，而 label 只有一个维度，所以需要修改 pred 或 label 的大小。
    #     #label 是一个 GPS 坐标的标签，则需要将 label 的大小修改为 (16, 2)，以匹配 pred 的大小。
    #     pred = self.forward(traj_h)
    #     if self.pred_type == 'regression':
    #         if denormalize: # GPS预测值的反归一化
    #             pred = pred * (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0) + \
    #                     self.spatial_border[0].unsqueeze(0)
    #         # 将 label 的大小修改为 (16, 2)
    #         if tte_loss:
    #             label = label.unsqueeze(-1).repeat(1, 2)
    #         loss = F.mse_loss(pred, label)
    #     elif self.pred_type == 'classification':
    #         loss = F.cross_entropy(pred, label.long().squeeze(-1))
    #     else:
    #         raise NotImplementedError(f'No prediction type: {self.pred_type}.')

    #     return loss

    def loss(self, traj_h, label, denormalize=False):
        pred = self.forward(traj_h)
        #tte
        # label = label.unsqueeze(1)  # 将标签的维度从 (16,) 改为 (16, 1)
        # label = label.repeat(1, 2)  # 将标签重复成 (16, 2)

        if self.pred_type == 'regression':
            if denormalize: # GPS预测值的反归一化
                pred = pred * (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0) + \
                        self.spatial_border[0].unsqueeze(0)
            loss = F.mse_loss(pred, label)
            
        elif self.pred_type == 'classification':
            loss = F.cross_entropy(pred, label.long().squeeze(-1))
        else:
            raise NotImplementedError(f'No prediction type: {self.pred_type}.')

        return loss

