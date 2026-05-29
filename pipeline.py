import numpy as np
import torch
from tqdm import trange, tqdm
from data import TrajectorySearchTestdata

def pretrain_model(model, dataloader, num_epoch, lr):
    """
    使用给定的训练数据加载器对模型进行预训练。

    参数:
        model (nn.Module): 要训练的模型。
        dataloader (DataLoader): 包含训练数据的批次迭代器。
        num_epoch (int): 训练的轮数。
        lr (float): 优化器的学习率。
    """
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # 进度条描述
    bar_desc = 'Pretraining, avg loss: %.5f'
    with trange(num_epoch, desc=bar_desc % 0.0, position=0) as bar:
        for epoch_i in bar:
            loss_values = []
            # 遍历数据集中的每个batch
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False, ncols=60):
                
                optimizer.zero_grad()  # 清除梯度
                loss = model.loss(*batch)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                loss_values.append(loss.item())  # 记录每个batch的损失
            bar.set_description(bar_desc % np.mean(loss_values))  # 更新进度条的平均损失


def finetune_model(model, pred_head, dataloader, num_epoch, lr, ft_encoder=True, denormalize=False):
    """
    使用特定任务标签对模型进行微调。

    参数:
        model (nn.Module): 要微调的模型。
        pred_head (nn.Module): 用于将嵌入映射到预测的预测头。
        dataloader (DataLoader): 包含微调数据的批次迭代器。
        num_epoch (int): 微调的轮数。
        lr (float): 优化器的学习率。
        ft_encoder (bool, optional): 是否微调轨迹编码器。默认为True。如果为False，则只微调任务特定的预测模块。
        denormalize (bool, optional): 是否对预测进行反归一化。默认为False。
    """
    pred_head.train()  # 设置预测头为训练模式
    if ft_encoder:
        # 如果微调编码器，则同时优化模型和预测头
        print('如果微调编码器，则同时优化模型和预测头')
        optimizer = torch.optim.Adam(list(model.parameters()) + list(pred_head.parameters()), lr=lr)
        model.train()
    else:
        # 如果不微调编码器，则只优化预测头
        print('如果不微调编码器，则只优化预测头')
        optimizer = torch.optim.Adam(pred_head.parameters(), lr=lr)
        model.eval()  # 冻结模型

    # 进度条描述
    bar_desc = 'Finetuning, avg loss: %.5f'
    with trange(num_epoch, desc=bar_desc % 0.0, position=0) as bar:
        for epoch_i in bar:
            loss_values = []
            # 遍历数据集中的每个batch
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False, ncols=60):
                *input_batch, label = batch  # 分离输入和标签
                optimizer.zero_grad()  # 清除梯度
                traj_h = model(*input_batch)  # 获取轨迹的嵌入
                if not ft_encoder:
                    traj_h = traj_h.detach()  # 如果不微调编码器，则不更新轨迹嵌入
                loss = pred_head.loss(traj_h, label, denormalize)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                loss_values.append(loss.item())  # 记录每个batch的损失
            bar.set_description(bar_desc % np.mean(loss_values))  # 更新进度条的平均损失


@torch.no_grad()
def test_model(model, pred_head, dataloader, denormalize=False):
    """
    使用特定预测任务测试模型。

    参数:
        model (nn.Module): 要测试的轨迹嵌入模型。
        pred_head (nn.Module): 用于将轨迹嵌入映射到预测的预测头。
        dataloader (DataLoader): 包含测试数据的批次迭代器。
        denormalize (bool, optional): 是否对预测进行反归一化。默认为False。
    
    返回:
        tuple: 预测结果和真实标签。
    """
    model.eval()  # 设置模型为评估模式
    pred_head.eval()  # 设置预测头为评估模式

    predictions, targets = [], []
    # 遍历测试数据集中的每个batch
    for batch in tqdm(dataloader, 'Testing', ncols=60):
        *input_batch, target = batch  # 分离输入和目标
        traj_h = model(*input_batch)  # 获取轨迹嵌入
        pred = pred_head(traj_h)  # 进行预测
        if denormalize:
            # 如果需要反归一化，则进行反归一化操作
            pred = pred * (pred_head.spatial_border[1] - pred_head.spatial_border[0]).unsqueeze(0) + \
                        pred_head.spatial_border[0].unsqueeze(0)
        predictions.append(pred.cpu().numpy())  # 将预测结果移动到CPU并转换为numpy数组
        targets.append(target.cpu().numpy())  # 将目标标签移动到CPU并转换为numpy数组
    predictions = np.concatenate(predictions, 0)  # 合并所有预测结果
    targets = np.concatenate(targets, 0)  # 合并所有真实标签
    return predictions, targets


@torch.no_grad()
def test_model_on_search(model, traj_dataloader, qrytgt_dataloader, neg_indices, set_name="test", tte_loss=False):
    """
    使用相似轨迹搜索测试模型。

    参数:
        model (nn.Module): 要测试的轨迹嵌入模型。
        traj_dataloader (DataLoader): 包含轨迹数据的批次迭代器。
        qrytgt_dataloader (DataLoader): 包含查询轨迹和目标轨迹的批次迭代器。
        neg_indices (ndarray): 负样本索引。
        set_name (str, optional): 数据集名称，默认为"test"。

    返回:
        tuple: 预测结果和真实标签。
    """
    model.eval()  # 设置模型为评估模式

    # 计算查询和目标轨迹的嵌入
    qrytgt_embeds = []
    for batch_meta in tqdm(qrytgt_dataloader,
                            desc=f"Calculating query and target embeds on {set_name} set",
                            total=len(qrytgt_dataloader), ncols=60):
        encodes = model(*batch_meta)
        qrytgt_embeds.append(encodes.detach().cpu().numpy())
    qrytgt_embeds = np.concatenate(qrytgt_embeds, 0)  # 合并查询和目标轨迹的嵌入
    qry_indices, tgt_indices = TrajectorySearchTestdata.parse_label(len(qrytgt_embeds))

    # 计算所有轨迹的嵌入
    embeds = []
    whole_enc_time = []
    traj_process_time = []
    for batch_meta in tqdm(traj_dataloader,
                            desc=f"Calculating embeds on {set_name} set",
                            total=len(traj_dataloader), ncols=60):
        encodes, enc_time, process_time = model.forward_on_search_mode(*batch_meta)
        embeds.append(encodes.detach().cpu().numpy())  # 获取轨迹嵌入
        whole_enc_time.append(enc_time)  # 记录编码时间
        traj_process_time.append(process_time)  # 记录轨迹处理时间
    whole_enc_time = np.array(whole_enc_time)  # 转换为numpy数组
    traj_process_time = np.array(traj_process_time)  # 转换为numpy数组
    print("Embedding time: {:.3f}s".format(whole_enc_time.sum()))  # 输出总编码时间
    print("Check traj process time: {:.3f}s".format(traj_process_time.sum()))  # 输出总轨迹处理时间
    embeds = np.concatenate(embeds, 0)  # 合并轨迹的嵌入

    # 计算预测结果和标签
    predictions, targets = TrajectorySearchTestdata.cal_pres_and_labels(qrytgt_embeds[qry_indices], qrytgt_embeds[tgt_indices], embeds[neg_indices])

    return predictions, targets
