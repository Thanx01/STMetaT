import os
import json
from argparse import ArgumentParser
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import OrderedDict
import higher
from torch.utils.data import Dataset
import copy

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
        
        for batch in dataloader:
            support_trajs, query_trajs = batch  # batch_size 个任务的支持集和查询集
            # support_trajs: (batch_size, k_shot, 5), query_trajs: (batch_size, q_shot, 5)
            support_trajs = support_trajs.to(device)
            query_trajs = query_trajs.to(device)
            support_lens = torch.tensor([support_trajs.shape[1]] * support_trajs.shape[0], device=device)  # 每个任务的长度
            query_lens = torch.tensor([query_trajs.shape[1]] * query_trajs.shape[0], device=device)
            
            # 内循环
            task_model = copy.deepcopy(model)
            task_optimizer = torch.optim.Adam(task_model.parameters(), lr=task_lr)
            for _ in range(num_shots):
                task_optimizer.zero_grad()
                loss_support = task_model.loss(support_trajs, support_lens)  # 直接传入批次数据
                loss_support.backward()
                task_optimizer.step()
            
            # 外循环
            loss_query = task_model.loss(query_trajs, query_lens)
            meta_train_error += loss_query.item()
            loss_query.backward()
            meta_optimizer.step()
            meta_optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{num_epochs}, Meta Train Error: {meta_train_error:.4f}")

def get_city_from_path(dataset_path):
    """
    从数据集路径中提取城市名称，假设城市名称为路径中包含的文件夹名称（例如 'chengdu' 或 'xian'）。
    """
    # 提取文件夹名称作为城市名
    city_name = dataset_path.split('/')[9]  # 根据路径层级调整索引
    return city_name    


class MetaTaskDataset(Dataset):
    """从 base_dataset 中按时间窗口和空间网格采样支持集和查询集，确保来自同一条轨迹。"""
    def __init__(self, base_dataset, num_tasks=10000, k_shot=5, q_shot=5, time_window=900, grid_size=0.005, device='cuda'):
        super().__init__()
        self.base_dataset = base_dataset
        self.num_tasks = num_tasks
        self.k_shot = k_shot
        self.q_shot = q_shot
        self.time_window = time_window  # 时间窗口大小，单位：秒，例如 900 秒
        self.grid_size = grid_size      # 空间网格大小，例如 0.005 度（约 1km）
        self.device = device

        # 预处理：按网格和时间窗口组织轨迹
        self.grid_time_trajs = self._organize_by_grid_and_time()

    def _organize_by_grid_and_time(self):
        """为每条轨迹分配网格 ID 和时间窗口。"""
        grid_time_trajs = {}
        min_lng, min_lat = self.base_dataset.spatial_border[0]

        for traj_id in self.base_dataset.traj_ids:
            traj = self.base_dataset.traj_df[self.base_dataset.traj_df[TRAJ_ID_COL] == traj_id].copy()
            # 计算 delta_t
            traj[DT_COL] = traj[T_COL] - traj[T_COL].iloc[0]
            # 确保道路列是数值型
            if traj[ROAD_COL].dtype not in [np.int64, np.float64]:
                traj[ROAD_COL] = pd.to_numeric(traj[ROAD_COL], errors='coerce').fillna(0).astype(np.int64)

            # 计算轨迹的平均经纬度
            avg_lng = traj[X_COL].mean()
            avg_lat = traj[Y_COL].mean()
            grid_x = int((avg_lng - min_lng) // self.grid_size)
            grid_y = int((avg_lat - min_lat) // self.grid_size)
            grid_id = (grid_x, grid_y)

            # 按时间窗口分组轨迹
            traj['time_window'] = (traj[T_COL] // self.time_window).astype(int)
            for window_key, group in traj.groupby('time_window'):
                key = (grid_id, window_key)
                if key not in grid_time_trajs:
                    grid_time_trajs[key] = []
                grid_time_trajs[key].append({
                    'traj_id': traj_id,
                    'points': group[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy(dtype=np.float64)
                })

        return grid_time_trajs

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        """返回一个任务的支持集和查询集，来自同一网格和时间窗口的同一条轨迹。"""
        # 随机选择一个网格和时间窗口
        keys = list(self.grid_time_trajs.keys())
        selected_key = keys[np.random.randint(len(keys))]
        trajs_in_key = self.grid_time_trajs[selected_key]

        # 随机选择一条轨迹
        selected_traj = trajs_in_key[np.random.randint(len(trajs_in_key))]
        traj_points = selected_traj['points']

        # 确保轨迹点数足够
        while len(traj_points) < (self.k_shot + self.q_shot):
            selected_key = keys[np.random.randint(len(keys))]
            trajs_in_key = self.grid_time_trajs[selected_key]
            selected_traj = trajs_in_key[np.random.randint(len(trajs_in_key))]
            traj_points = selected_traj['points']

        # 随机采样 k_shot + q_shot 个点
        sampled_indices = np.random.choice(len(traj_points), size=self.k_shot + self.q_shot, replace=False)
        support_indices = sampled_indices[:self.k_shot]
        query_indices = sampled_indices[self.k_shot:]

        support_array = traj_points[support_indices]
        query_array = traj_points[query_indices]

        support_tensor = torch.from_numpy(support_array).float().to(self.device)
        query_tensor = torch.from_numpy(query_array).float().to(self.device)

        return support_tensor, query_tensor


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
        # pred_head = MlpPredictor(spatial_border=train_dataset.spatial_border, **setting['pred_head']).to(device)
        pred_head = MlpPredictor(spatial_border=test_dataset.spatial_border, **setting['pred_head']).to(device)
        size_all_mb = utils.cal_model_size(traj_clip.traj_view)
        print(f"Trajectory-Mamba Model size: {size_all_mb} MBytes.")


        # 检查设置中是否包含预训练配置
        if 'pretrain' in setting:
            print("time_window = 2700")
            print(parser.parse_args())
            # Pretrain the trajectory embedding model with self-supervised CLIP loss.
            # if setting['pretrain'].get('load', False):
            #     # Load previously saved model parameters.
            #     PRETRAIN_SAVE_NAME = setting['pretrain'].get('pretrain_save_name', SAVE_NAME)
            #     traj_clip.load_state_dict(
            #         torch.load(os.path.join(MODEL_CACHE_DIR, f'{PRETRAIN_SAVE_NAME}.pretrain'), map_location=device)
            #     )
            # else:
                # if setting['pretrain'].get('maml', False):  # MAML模式
                    # 创建元任务数据集
                    #成都时间步长为900秒，然而西安需要另外设置时间步长time_window，由于西安轨迹数据集更稀疏，所以时间步长设置为2700秒
                    # 获取城市名称  
            city_name = get_city_from_path(setting['dataset']['test_traj_df'])  # 提取城市名称
            print("city_name:", city_name)
            time_window = 2700 if city_name == 'chengdu' else 2700  # 成都 900 秒，西安 5400 秒
            meta_dataset = MetaTaskDataset(
                base_dataset=train_dataset,
                num_tasks=10000,  # 子任务数量
                k_shot=setting['pretrain']['config']['k_shot'],
                q_shot=setting['pretrain']['config']['q_shot'],
                time_window=time_window,  # 15分钟 ,新增时间窗口参数
                grid_size=0.005,  # 约 5km
                device=device
            )

            pretrain_dataloader = DataLoader(
                meta_dataset,
                batch_size=4096,  # 修改为 32 个任务，确保批次内有多轨迹数据
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
            if setting['pretrain'].get('save', True):
                # Save the pretrained model parameters with datetime_key in filename
                utils.create_if_noexists(MODEL_CACHE_DIR)
                PRETRAIN_SAVE_NAME = f"{SAVE_NAME}_pretrain_{datetime_key}_Ptrajm_maml_test_grid_size=0.005"
                torch.save(traj_clip.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{PRETRAIN_SAVE_NAME}.pretrain'))
                # else:  # 原始CLIP预训练
                    # # 创建预训练数据加载器
                    # pretrain_dataloader = DataLoader(
                    #     train_dataset,
                    #     collate_fn=PretrainPadder(
                    #         device=device, **setting['pretrain']['padder']
                    #     ),
                    #     **setting['pretrain']['dataloader']
                    # )
                    # pretrain_model(model=traj_clip, dataloader=pretrain_dataloader, **setting['pretrain']['config'])
                    
                    # # 检查是否需要保存预训练模型的参数
                    # if setting['pretrain'].get('save', True):
                    #     utils.create_if_noexists(MODEL_CACHE_DIR)
                    #     torch.save(traj_clip.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}.pretrain'))                        

        if 'finetune' in setting:
            # Finetune the trajectory embedding model and the prediction head on downstream tasks.
            print('开始微调')
            # if setting['finetune'].get('load', False):
            #     traj_clip.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_trajclip.finetune')))
            #     pred_head.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_predhead.finetune')))
            # else:
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
                FINETUNE_SAVE_NAME = f"{SAVE_NAME}_finetune_{datetime_key}_Ptrajm_test_grid_size=0.005"
                torch.save(traj_clip.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{FINETUNE_SAVE_NAME}_trajclip.finetune'))
                torch.save(pred_head.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{FINETUNE_SAVE_NAME}_predhead.finetune'))

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
                print(traj_clip)
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
                # 根据城市设定时间窗口（成都900秒、西安2700秒）
                
                # 获取城市名称
                city_name = get_city_from_path(setting['dataset']['test_traj_df'])  # 提取城市名称

                time_window = 2700 if city_name == 'chengdu' else 2700
                grid_size = 0.005  # 你代码中定义的固定网格大小
                batch_size = setting['test']['dataloader']['batch_size']


                # 构造保存路径
                save_dir = os.path.join(
                    PRED_SAVE_DIR, 'meta', city_name, SAVE_NAME, down_task
                )
                utils.create_if_noexists(save_dir)

                # 文件名前缀
                save_file_prefix = f"{city_name}_tw{time_window}s_grid{grid_size}_bs{batch_size}"
                
                print("city_name:", city_name)
                print(f"city: {city_name}, time_window: {time_window}, grid_size: {grid_size}, batch_size: {batch_size}")

                # 创建以城市为基础的保存目录
                city_dir = os.path.join(PRED_SAVE_DIR,'meta', city_name, SAVE_NAME, down_task)  # 创建城市文件夹 + 任务文件夹
                print("city_dir:", city_dir)

                # 确保目录存在
                utils.create_if_noexists(city_dir)

                # 保存预测和真实目标文件
                np.save(os.path.join(city_dir, f'{save_file_prefix}_predictions.npy'), predictions)
                np.save(os.path.join(city_dir, f'{save_file_prefix}_targets.npy'), targets)
                print(f"文件已保存至：{city_dir}")
                print(f"预测文件名：{save_file_prefix}_predictions.npy")
                print(f"目标文件名：{save_file_prefix}_targets.npy")
                if down_task == "search":
                    metric.to_hdf(os.path.join(city_dir, 'similar_trajectory_search.h5'), key='metric', format='table')
                
if __name__ == '__main__':
    main()
