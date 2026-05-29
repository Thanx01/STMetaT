import os
import json
from argparse import ArgumentParser
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import OrderedDict
import higher
from torch.utils.data import Dataset
parser = ArgumentParser()
parser.add_argument('-s', '--settings', help='name of the settings file to use', type=str, default="local_test") # required=True
parser.add_argument('--cuda', help='index of the cuda device to use', type=int, default='1')
args = parser.parse_args()
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '0'

import torch
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
import utils
from data import TrajClipDataset, PretrainPadder, DpPadder, TtePadder, TrajectorySearchTestdata, TrajectorySearchDataset, SearchPadder, fetch_task_padder, load_trajSearch_testdata, X_COL, Y_COL, SEARCH_META_DIR
from pipeline import pretrain_model, finetune_model, test_model, test_model_on_search
from models.traj_clip import TrajClip
from models.predictor import MlpPredictor

# Parsing arguments and setting environment
parser = ArgumentParser()
parser.add_argument('-s', '--settings', help='name of the settings file to use', type=str, default="local_test")  # required=True
parser.add_argument('--cuda', help='index of the cuda device to use', type=int, default='1')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '0'

SETTINGS_CACHE_DIR = os.environ.get('SETTINGS_CACHE_DIR', os.path.join('settings', 'cache'))
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', 'saved_model')
PRED_SAVE_DIR = os.environ.get('PRED_SAVE_DIR', 'predictions')
TRAJ_ID_COL = 'trip'
X_COL = 'lng'
Y_COL = 'lat'
T_COL = 'timestamp'
DT_COL = 'delta_t'
ROAD_COL = 'road'
COL_I = {
    "spatial": [0, 1],
    "temporal": [2, 3],
    "road": 4
}
FEATURE_PAD = 0
MIN_TRIP_LEN = 5
MAX_TRIP_LEN = 120



def maml_train(model, dataloader, num_epochs, meta_lr, task_lr, device, num_shots):
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    for epoch in range(num_epochs):
        model.train()
        meta_train_error = 0.0
        
        for task_idx, (support_traj, query_traj) in enumerate(dataloader):
            support_traj = support_traj.to(device)   # shape (1,k_shot,5)
            query_traj   = query_traj.to(device)     # shape (1,q_shot,5)

            support_lens = torch.tensor([support_traj.shape[1]], device=device)
            query_lens   = torch.tensor([query_traj.shape[1]], device=device)
            # 1) 克隆一份 model => task_model
            task_model = copy.deepcopy(model)
            task_optimizer = torch.optim.Adam(task_model.parameters(), lr=task_lr)

            # 2) 内循环 (支持集)
            for _ in range(num_shots):
                task_optimizer.zero_grad()
                loss_support = task_model.loss(support_traj, support_lens)
                loss_support.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), 5.0)
                task_optimizer.step()

            # 3) 外循环 (查询集)
            loss_query = task_model.loss(query_traj, query_lens)
            meta_train_error += loss_query.item()
            
            meta_optimizer.zero_grad()
            loss_query.backward()
            # 再做一次梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            meta_optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Meta Train Error: {meta_train_error:.4f}")
        
def get_city_from_path(dataset_path):
    """
    从数据集路径中提取城市名称，假设城市名称为路径中包含的文件夹名称（例如 'chengdu' 或 'xian'）。
    """
    # 提取文件夹名称作为城市名
    city_name = dataset_path.split('/')[9]  # 根据路径层级调整索引
    return city_name    


# 原始预训练中的 Padder
# PretrainPadder：
# 作用：将批次中长度不一的轨迹填充到相同长度，返回 (traj_batch, valid_lens)。
# 必要性：原始预训练处理完整轨迹，长度不固定，需要填充以支持批处理。
# 元学习中的情况
# 支持集和查询集长度：
# 支持集长度固定为 k_shot，查询集长度固定为 q_shot。
# 在 MetaTaskDataset 中已直接生成固定长度的张量。
# 当前实现：
# __getitem__ 返回的 (support_tensor, query_tensor) 已转换为张量并移到设备上。
# DataLoader 无需额外的 collate_fn 来填充长度。
# 建议
# 不需要额外的 Padder：
# 因为支持集和查询集的长度是固定的（k_shot 和 q_shot），无需动态填充。
# 当前的实现已经足够高效。
# 如果需要 Padder 的场景：
# 如果你修改采样方式，导致支持集和查询集长度不固定（例如，基于时间窗口采样返回变长数据），才需要编写类似 PretrainPadder 的方法。
# 但根据你的现有需求和建议的优化方向（随机不连续采样），长度仍是固定的，因此无需 Padder。

class MetaTaskDataset(Dataset):
    """从 base_dataset 中随机抽取 num_tasks 个任务，每个任务都有 k_shot 和 q_shot。"""
    def __init__(self, base_dataset, num_tasks=10000, k_shot=5, q_shot=5, device='cuda'):
        super().__init__()
        self.base_dataset = base_dataset
        self.num_tasks = num_tasks
        self.k_shot = k_shot
        self.q_shot = q_shot
        self.device = device

        # 获取所有轨迹 ID，或其他所需信息
        self.all_traj_ids = base_dataset.traj_ids

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        """
        返回第 index 个任务的 (support_traj, query_traj)，
        都是 torch.Tensor 已经放到 device 上了
        """
        one_traj = self._sample_one_traj_from_base()
        # 假设 one_traj 是一个 pd.DataFrame 或 numpy array
        # 这里要先把它变成 numpy，然后再转到 torch.Tensor
        # 并切分 k_shot / q_shot
        
        raw_array = one_traj[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()
        while True:
            one_traj = self._sample_one_traj_from_base()
            raw_array = one_traj[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()
            if raw_array.shape[0] < (self.k_shot + self.q_shot):
                # 说明这条轨迹不够长，跳过，继续while循环重新random_idx
                continue
            else:
                # 足够长，正常处理
                break

        support_array = raw_array[:self.k_shot]
        query_array = raw_array[self.k_shot : self.k_shot + self.q_shot]
        # 转成 Tensor
        support_tensor = torch.from_numpy(support_array).float().to(self.device)
        query_tensor = torch.from_numpy(query_array).float().to(self.device)
        
        return support_tensor, query_tensor

    def _sample_one_traj_from_base(self):
        random_idx = np.random.randint(0, len(self.base_dataset))
        return self.base_dataset[random_idx]


def main():
    device = f'cuda:0' if torch.cuda.is_available() and args.cuda is not None else 'cpu'

    # This key is an indicator of multiple things.
    datetime_key = utils.get_datetime_key()
    print(f'====START EXPERIMENT, DATETIME KEY: {datetime_key} ====')

    # Load the settings file, and save a backup in the cache directory.
    with open(os.path.join('settings', f'{args.settings}.json'), 'r') as fp:
        settings = json.load(fp)
    utils.create_if_noexists(SETTINGS_CACHE_DIR)
    with open(os.path.join(SETTINGS_CACHE_DIR, f'{datetime_key}.json'), 'w') as fp:
        json.dump(settings, fp)

    # Iterate through the multiple settings.加载并构建训练和测试数据集。
    for setting_i, setting in enumerate(settings):
        print(f'===SETTING {setting_i}/{len(settings)}===')
        SAVE_NAME = setting.get('save_name', None)

        # Load and build training and testing datasets.
        train_traj_df = pd.read_hdf(setting['dataset']['train_traj_df'], key='trips')
        test_traj_df = pd.read_hdf(setting['dataset']['test_traj_df'], key='trips')
        train_dataset = TrajClipDataset(traj_df=train_traj_df)
        test_dataset = TrajClipDataset(traj_df=test_traj_df)

        # Load road segments and POIs' coordinates and textual embeddings.加载路段和 POI 的坐标和文本嵌入。
        road_embed = np.load(setting['dataset']['road_embed'])
        poi_df = pd.read_hdf(setting['dataset']['poi_df'], key='pois')
        poi_embed = np.load(setting['dataset']['poi_embed'])
        poi_coors = poi_df[[X_COL, Y_COL]].to_numpy()

        # Build the trajectory embedding model and the downstream prediction head.
        traj_clip = TrajClip(road_embed=road_embed, poi_embed=poi_embed, poi_coors=poi_coors,
                             spatial_border=train_dataset.spatial_border, device=device, **setting['traj_clip']).to(device)
        pred_head = MlpPredictor(spatial_border=train_dataset.spatial_border, **setting['pred_head']).to(device)
        size_all_mb = utils.cal_model_size(traj_clip.traj_view)
        print(f"Trajectory-Mamba Model size: {size_all_mb} MBytes.")

        # 检查设置中是否包含预训练配置
        if 'pretrain' in setting:
            # Pretrain the trajectory embedding model with self-supervised CLIP loss.
            if setting['pretrain'].get('load', False):
                # Load previously saved model parameters.
                PRETRAIN_SAVE_NAME = setting['pretrain'].get('pretrain_save_name', SAVE_NAME)
                traj_clip.load_state_dict(
                    torch.load(os.path.join(MODEL_CACHE_DIR, f'{PRETRAIN_SAVE_NAME}.pretrain'), map_location=device)
                )
            else:
                if setting['pretrain'].get('maml', False):  # MAML模式
                    # 创建元任务数据集
                    meta_dataset = MetaTaskDataset(
                        base_dataset=train_dataset,
                        num_tasks=50,       # 子任务数量
                        k_shot=setting['pretrain']['config']['k_shot'],
                        q_shot=setting['pretrain']['config']['q_shot'],
                        device=device
                    )

                    pretrain_dataloader = DataLoader(
                        meta_dataset,
                        batch_size=10000,        # 每次只取一个任务，也可以大于1
                        shuffle=True,
                        num_workers=0
                    )
                    maml_train(
                        model=traj_clip,
                        dataloader=pretrain_dataloader,
                        num_epochs=setting['pretrain']['config']['num_epoch'],
                        meta_lr=setting['pretrain']['config']['meta_lr'],
                        task_lr=setting['pretrain']['config']['inner_lr'],
                        device=device,
                        num_shots=setting['pretrain']['config']['k_shot']
                    )



                else:  # 原始CLIP预训练
                    # 创建预训练数据加载器
                    pretrain_dataloader = DataLoader(
                        train_dataset,
                        collate_fn=PretrainPadder(
                            device=device, **setting['pretrain']['padder']
                        ),
                        **setting['pretrain']['dataloader']
                    )
                    pretrain_model(model=traj_clip, dataloader=pretrain_dataloader, **setting['pretrain']['config'])
                    
                    # 检查是否需要保存预训练模型的参数
                    if setting['pretrain'].get('save', True):
                        utils.create_if_noexists(MODEL_CACHE_DIR)
                        torch.save(traj_clip.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}.pretrain'))
                                
                        
        if 'finetune' in setting:
            # Finetune the trajectory embedding model and the prediction head on downstream tasks.
            print('开始微调')
            if setting['finetune'].get('load', False):
                traj_clip.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_trajclip.finetune')))
                pred_head.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_predhead.finetune')))
            else:
                finetune_padder = fetch_task_padder(padder_name=setting['finetune']['padder']['name'],
                                                    device=device, padder_params=setting['finetune']['padder']['params'])
                finetune_dataloader = DataLoader(train_dataset, collate_fn=finetune_padder,
                                                 **setting['finetune']['dataloader'])
                if_denormalize = False
                if isinstance(finetune_padder, DpPadder):
                    if sorted(finetune_padder.pred_cols) == sorted([Y_COL, X_COL]): # need to denormalize predictor optput
                        if_denormalize = True
                finetune_model(model=traj_clip, pred_head=pred_head, dataloader=finetune_dataloader, denormalize=if_denormalize,
                               **setting['finetune']['config'])

                if setting['finetune'].get('save', True):
                    utils.create_if_noexists(MODEL_CACHE_DIR)
                    torch.save(traj_clip.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_trajclip.finetune'))
                    torch.save(pred_head.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_predhead.finetune'))

        if 'test' in setting:

            #在这里修改下游任务
            
            down_task = setting['test'].get('task', "tte")
            if down_task == "dp":
                print('dp任务')
                test_padder = fetch_task_padder(padder_name=setting['test']['padder']['name'],
                                                device=device, padder_params=setting['test']['padder']['params'])
                test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=test_padder,
                                            **setting['test']['dataloader'])
                if_denormalize = False
                if isinstance(test_padder, DpPadder):
                    print('DpPadder is instance')
                    if sorted(test_padder.pred_cols) == sorted([Y_COL, X_COL]):
                        if_denormalize = True
                predictions, targets = test_model(model=traj_clip, pred_head=pred_head, dataloader=test_dataloader, denormalize=if_denormalize)
            
            elif down_task == "search":
                print('search任务')
                eval_dataset = os.path.basename(setting['dataset']['test_traj_df']).split(".")[0]
                search_meta_dir = os.path.join(SEARCH_META_DIR, eval_dataset)
                try:
                    alltrajtgt, hopqrytgt, neg_indices = load_trajSearch_testdata(search_meta_dir, **setting['test']["search_data_params"])
                except FileNotFoundError:
                    print("Generate meta for similar trajectory search and Save")
                    simTrajSearch_testData = TrajectorySearchTestdata(test_dataset, spatial_border=train_dataset.spatial_border, **setting['test']["search_data_params"])
                    simTrajSearch_testData.save_search_meta(search_meta_dir)
                    alltrajtgt, hopqrytgt, neg_indices = simTrajSearch_testData.get_search_meta()
                
                alltrajtgt_dataset = TrajectorySearchDataset(alltrajtgt)
                trajqrytgt_dataset = TrajectorySearchDataset(hopqrytgt)
                alltrajtgt_dataloader = DataLoader(alltrajtgt_dataset, shuffle=False, collate_fn=SearchPadder(device=device),
                                                **setting['test']['dataloader'])
                trajqrytgt_dataloader = DataLoader(trajqrytgt_dataset, shuffle=False, collate_fn=SearchPadder(device=device),
                                                **setting['test']['dataloader'])
                predictions, targets = test_model_on_search(model=traj_clip, traj_dataloader=alltrajtgt_dataloader, qrytgt_dataloader=trajqrytgt_dataloader, neg_indices=neg_indices)
                metric = utils.cal_classification_metric(targets, predictions)
                metric["mean_rank"] = utils.cal_mean_rank(predictions, targets)
                print(f"the test metric for similar trajectory search:")
                print(metric)

            elif down_task == "tte":
                print('tte任务')
                test_padder = fetch_task_padder(padder_name=setting['test']['padder']['name'],
                                                device=device, padder_params=setting['test']['padder']['params'])
                test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=test_padder,
                                            **setting['test']['dataloader'])
                if_denormalize = False
                predictions, targets = test_model(model=traj_clip, pred_head=pred_head, dataloader=test_dataloader, denormalize=if_denormalize)
            
            else:
                raise NotImplementedError(f'No downstream task called "{down_task}".')

            if setting['test'].get('save', False):
                # 获取城市名称
                city_name = get_city_from_path(setting['dataset']['test_traj_df'])  # 提取城市名称
                print("city_name:", city_name)

                # 创建以城市为基础的保存目录
                city_dir = os.path.join(PRED_SAVE_DIR,'meta', city_name, SAVE_NAME, down_task)  # 创建城市文件夹 + 任务文件夹
                print("city_dir:", city_dir)

                # 确保目录存在
                utils.create_if_noexists(city_dir)

                # 保存预测和真实目标文件
                np.save(os.path.join(city_dir, 'predictions.npy'), predictions)
                np.save(os.path.join(city_dir, 'targets.npy'), targets)
                if down_task == "search":
                    metric.to_hdf(os.path.join(city_dir, 'similar_trajectory_search.h5'), key='metric', format='table')
                
if __name__ == '__main__':
    main()
    
# [
#     {
#         "默认pretrain_batch_size":"16",
#         "默认finetune_batch_size":"16",
#         "默认finetune_epoch":"30",
#         "默认pretrain_epoch":"5",
#         "save_name": "local_test",
#         "dataset": {
#             "train_traj_df": "/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/datasets_from_Lin/chengdu/chengdu.h5",
#             "test_traj_df": "/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/datasets_from_Lin/chengdu/chengdu.h5",
#             "poi_df": "/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/datasets_from_Lin/chengdu/chengdu.h5",
#             "poi_embed": "./samples/small_chengdu_poi_embed.npy",
#             "road_embed": "./samples/small_chengdu_road_embed.npy"
#         },
#         "traj_clip": {
#             "embed_size": 64,
#             "d_model": 128
#         },
#         "pred_head": {
#             "input_size": 128,
#             "hidden_size": 256,
#             "output_size": 2,
#             "pred_type": "regression"
#         },
#         "pretrain": {
#             "load": false,
#             "save": true,
#             "dataloader": {
#                 "batch_size": 16,
#                 "shuffle": true,
#                 "num_workers": 0
#             },
#             "maml": true, 
#             "padder": {},
#             "config": {
#                 "lr": 1e-3,
#                 "num_epoch": 200,
#                 "meta_lr": 1e-4,  
#                 "inner_lr": 1e-4, 
#                 "num_inner_steps": 3, 
#                 "k_shot": 10,      
#                 "q_shot": 10       
#             }
#         },
#         "finetune": {
#             "load": false,
#             "save": true,
#             "dataloader": {
#                 "batch_size": 16,
#                 "shuffle": true,
#                 "num_workers": 0
#             },
#             "padder": {
#                 "name": "dp",
#                 "params": {
#                     "pred_len": 5,
#                     "pred_cols": [
#                         "lng",
#                         "lat"
#                     ]
#                 }
#             },
#             "config": {
#                 "num_epoch": 30,
#                 "lr": 1e-3,
#                 "ft_encoder": true
#             }
#         },
#         "test": {
#             "task": "dp",
#             "save": true,
#             "dataloader": {
#                 "batch_size": 16,
#                 "num_workers": 0
#             },
#             "padder": {
#                 "name": "dp",
#                 "params": {
#                     "pred_len": 5,
#                     "pred_cols": [
#                         "lng",
#                         "lat"
#                     ]
#                 }
#             },
#             "search_data_params": {
#                 "num_target": 1000,
#                 "num_negative": 5000,
#                 "neg_random_choice": false
#             }
#         }
#     }
# ]