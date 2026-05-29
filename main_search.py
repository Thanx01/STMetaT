import os
import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd

parser = ArgumentParser()
parser.add_argument('-s', '--settings', help='name of the settings file to use', type=str, default="local_test") # required=True
parser.add_argument('--cuda', help='index of the cuda device to use', type=int, default='2')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '0'

import torch
from torch.utils.data import DataLoader

import utils
from data import TrajClipDataset, PretrainPadder, DpPadder, TtePadder, TrajectorySearchTestdata, TrajectorySearchDataset, SearchPadder, fetch_task_padder, load_trajSearch_testdata, X_COL, Y_COL, SEARCH_META_DIR
from pipeline import pretrain_model, finetune_model, test_model, test_model_on_search
from models.traj_clip import TrajClip
from models.predictor import MlpPredictor


SETTINGS_CACHE_DIR = os.environ.get('SETTINGS_CACHE_DIR', os.path.join('settings', 'cache'))
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', 'saved_model')
PRED_SAVE_DIR = os.environ.get('PRED_SAVE_DIR', 'predictions')

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

        # 检查设置中是否包含预训练配置
        if 'pretrain' in setting:
            # Pretrain the trajectory embedding model with self-supervised CLIP loss.预训练轨迹嵌入模型使用自监督的 CLIP 损失函数
            if setting['pretrain'].get('load', False):
                # Load previously saved model parameters. 加载之前保存的模型参数
                # 如果设置中没有指定预训练模型的保存名称，则使用默认的保存名称
                PRETRAIN_SAVE_NAME = setting['pretrain'].get('pretrain_save_name', SAVE_NAME) # one pretrained model may correspond to multiple types of finetune. 
                # 注意：一个预训练模型可能对应多种类型的微调，因此需要指定预训练模型的保存名称
                traj_clip.load_state_dict(# 加载预训练模型的参数,指定预训练模型的文件路径,指定模型参数的加载位置（设备）
                                          torch.load(os.path.join(MODEL_CACHE_DIR, f'{PRETRAIN_SAVE_NAME}.pretrain'),
                                                     map_location=device))
            else:
                # 创建预训练数据加载器, 指定预训练数据加载器
                pretrain_dataloader = DataLoader(train_dataset,
                                                collate_fn=PretrainPadder(
                                                    device=device, **setting['pretrain']['padder']),
                                                **setting['pretrain']['dataloader'])
                pretrain_model(model=traj_clip, dataloader=pretrain_dataloader, **setting['pretrain']['config'])
                # 检查是否需要保存预训练模型的参数
                if setting['pretrain'].get('save', True):
                    # Save the pretrained model parameters.模型缓存目录（如果不存在）,
                    utils.create_if_noexists(MODEL_CACHE_DIR)
                    #获取预训练模型的参数
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
            
            down_task = setting['test'].get('task', "search")
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
                # # Create directory for saving results
                # task_dir = os.path.join(PRED_SAVE_DIR, SAVE_NAME, down_task)  # Create a directory for each task
                # utils.create_if_noexists(task_dir)
                
                # # Save predictions and targets with task-specific file names
                # np.save(os.path.join(task_dir, 'predictions.npy'), predictions)
                # np.save(os.path.join(task_dir, 'targets.npy'), targets)
                
                # 获取城市名称
                city_name = get_city_from_path(setting['dataset']['test_traj_df'])  # 提取城市名称
                print("city_name:", city_name)

                # 创建以城市为基础的保存目录
                city_dir = os.path.join(PRED_SAVE_DIR, city_name, SAVE_NAME, down_task)  # 创建城市文件夹 + 任务文件夹
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
