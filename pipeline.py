import numpy as np
import torch
from tqdm import trange, tqdm
from data import TrajectorySearchTestdata

def pretrain_model(model, dataloader, num_epoch, lr):
    """
    Pre-train the model with the given dataloader.

    Args:
        model (nn.Module): model to train.
        dataloader (DataLoader): training batch iterator.
        num_epoch (int): number of training epochs.
        lr (float): optimizer learning rate.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    bar_desc = 'Pretraining, avg loss: %.5f'
    with trange(num_epoch, desc=bar_desc % 0.0, position=0) as bar:
        for epoch_i in bar:
            loss_values = []
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False, ncols=60):
                
                optimizer.zero_grad()
                loss = model.loss(*batch)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
            bar.set_description(bar_desc % np.mean(loss_values))


def finetune_model(model, pred_head, dataloader, num_epoch, lr, ft_encoder=True, denormalize=False):
    """
    Fine-tune the model with task-specific labels.

    Args:
        model (nn.Module): trajectory encoder to fine-tune.
        pred_head (nn.Module): prediction head that maps embeddings to task outputs.
        dataloader (DataLoader): fine-tuning batch iterator.
        num_epoch (int): number of fine-tuning epochs.
        lr (float): optimizer learning rate.
        ft_encoder (bool, optional): whether to fine-tune the trajectory encoder.
        denormalize (bool, optional): whether to denormalize predictions.
    """
    pred_head.train()
    if ft_encoder:
        print('Fine-tuning encoder and prediction head')
        optimizer = torch.optim.Adam(list(model.parameters()) + list(pred_head.parameters()), lr=lr)
        model.train()
    else:
        print('Fine-tuning prediction head only')
        optimizer = torch.optim.Adam(pred_head.parameters(), lr=lr)
        model.eval()

    bar_desc = 'Finetuning, avg loss: %.5f'
    with trange(num_epoch, desc=bar_desc % 0.0, position=0) as bar:
        for epoch_i in bar:
            loss_values = []
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False, ncols=60):
                *input_batch, label = batch
                optimizer.zero_grad()
                traj_h = model(*input_batch)
                if not ft_encoder:
                    traj_h = traj_h.detach()
                loss = pred_head.loss(traj_h, label, denormalize)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
            bar.set_description(bar_desc % np.mean(loss_values))


@torch.no_grad()
def test_model(model, pred_head, dataloader, denormalize=False):
    """
    Test the model on a prediction task.

    Args:
        model (nn.Module): trajectory embedding model.
        pred_head (nn.Module): prediction head.
        dataloader (DataLoader): test batch iterator.
        denormalize (bool, optional): whether to denormalize predictions.
    
    Returns:
        tuple: predictions and ground-truth targets.
    """
    model.eval()
    pred_head.eval()

    predictions, targets = [], []
    for batch in tqdm(dataloader, 'Testing', ncols=60):
        *input_batch, target = batch
        traj_h = model(*input_batch)
        pred = pred_head(traj_h)
        if denormalize:
            pred = pred * (pred_head.spatial_border[1] - pred_head.spatial_border[0]).unsqueeze(0) + \
                        pred_head.spatial_border[0].unsqueeze(0)
        predictions.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
    predictions = np.concatenate(predictions, 0)
    targets = np.concatenate(targets, 0)
    return predictions, targets


@torch.no_grad()
def test_model_on_search(model, traj_dataloader, qrytgt_dataloader, neg_indices, set_name="test", tte_loss=False):
    """
    Test the model on similar trajectory search.

    Args:
        model (nn.Module): trajectory embedding model.
        traj_dataloader (DataLoader): trajectory batch iterator.
        qrytgt_dataloader (DataLoader): query-target trajectory batch iterator.
        neg_indices (ndarray): negative sample indices.
        set_name (str, optional): dataset split name.

    Returns:
        tuple: predictions and ground-truth targets.
    """
    model.eval()

    qrytgt_embeds = []
    for batch_meta in tqdm(qrytgt_dataloader,
                            desc=f"Calculating query and target embeds on {set_name} set",
                            total=len(qrytgt_dataloader), ncols=60):
        encodes = model(*batch_meta)
        qrytgt_embeds.append(encodes.detach().cpu().numpy())
    qrytgt_embeds = np.concatenate(qrytgt_embeds, 0)
    qry_indices, tgt_indices = TrajectorySearchTestdata.parse_label(len(qrytgt_embeds))

    embeds = []
    whole_enc_time = []
    traj_process_time = []
    for batch_meta in tqdm(traj_dataloader,
                            desc=f"Calculating embeds on {set_name} set",
                            total=len(traj_dataloader), ncols=60):
        encodes, enc_time, process_time = model.forward_on_search_mode(*batch_meta)
        embeds.append(encodes.detach().cpu().numpy())
        whole_enc_time.append(enc_time)
        traj_process_time.append(process_time)
    whole_enc_time = np.array(whole_enc_time)
    traj_process_time = np.array(traj_process_time)
    print("Embedding time: {:.3f}s".format(whole_enc_time.sum()))
    print("Trajectory processing time: {:.3f}s".format(traj_process_time.sum()))
    embeds = np.concatenate(embeds, 0)

    predictions, targets = TrajectorySearchTestdata.cal_pres_and_labels(qrytgt_embeds[qry_indices], qrytgt_embeds[tgt_indices], embeds[neg_indices])

    return predictions, targets
