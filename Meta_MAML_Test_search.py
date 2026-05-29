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


def get_city_from_path(dataset_path):
    """
    从数据集路径中提取城市名称，假设城市名称为路径中包含的文件夹名称（例如 'chengdu' 或 'xian'）。
    """
    # 提取文件夹名称作为城市名
    city_name = dataset_path.split('/')[9]  # 根据路径层级调整索引
    return city_name    


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


        if 'test' in setting:

            # 加载预训练模型    
            pretrain_model_path='/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/saved_model_chengdu/local_test_finetune_D2025_03_17T20_22_56_TRHK_Ptrajm_test_grid_size=0.06_predhead.finetune'
            print(f"加载预训练模型: {pretrain_model_path}")
            traj_clip.load_state_dict(torch.load(pretrain_model_path, map_location=device))
            traj_clip.eval()  # 设置为评估模式
            # search去注释
            # predhead_path='/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/saved_model_chengdu/local_test_predhead.finetune'
            # pred_head.load_state_dict(torch.load(predhead_path, map_location=device))

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

            if down_task == "search":
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
                print()
                predictions, targets = test_model_on_search(model=traj_clip, traj_dataloader=alltrajtgt_dataloader, qrytgt_dataloader=trajqrytgt_dataloader, neg_indices=neg_indices)
                metric = utils.cal_classification_metric(targets, predictions)
                metric["mean_rank"] = utils.cal_mean_rank(predictions, targets)
                print(f"the test metric for similar trajectory search:")
                print(metric)
                
            elif down_task == "tte":
                print('tte任务')
            #     traj_clip.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_trajclip.finetune')))
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

                time_window = 2700 
                grid_size = 0.1  # 你代码中定义的固定网格大小
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
