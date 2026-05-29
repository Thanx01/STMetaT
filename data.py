import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from collections import Counter
from einops import repeat, rearrange
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.metrics.pairwise import euclidean_distances
import utils


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
SEARCH_META_DIR = os.environ.get('SEARCH_META_DIR', os.path.join('processed_data', 'search_meta'))

class TrajClipDataset(Dataset):
    """
    Dataset support class for TrajCLIP.

    Args:
        traj_df (pd.DataFrame): contains points of all trajectories.
        traj_ids (pd.Series): records the unique IDs of all trajectory sequences.
        spatial_border (list): coordinates indicating the spatial border: [[x_min, y_min], [x_max, y_max]].
    """

    def __init__(self, traj_df):
        """
        Args:
            traj_df (pd.DataFrame): contains points of all trajectories.
        """
        super().__init__()
        # Filtering trips to keep the trajectories with length at [MIN_TRIP_LEN, MAX_TRIP_LEN]
        traj_ids = []
        # 遍历每个轨迹，通过groupby对`TRAJ_ID_COL`列进行分组处理
        # 如果轨迹的长度在 [MIN_TRIP_LEN, MAX_TRIP_LEN] 范围内且没有缺失值，就保留该轨迹
        for _, group in tqdm(traj_df.groupby(TRAJ_ID_COL), desc='Filtering trips', total=len(traj_df[TRAJ_ID_COL].unique()), leave=False, ncols=70):
            if (not group.isna().any().any()) and group.shape[0] >= MIN_TRIP_LEN and group.shape[0] <= MAX_TRIP_LEN:
                traj_ids.append(group.iloc[0]['trip'])
        
        # 保存符合条件的轨迹ID
        self.traj_ids = np.array(traj_ids)
        self.traj_df = traj_df[traj_df['trip'].isin(self.traj_ids)].copy()

        # 将时间列转换为时间戳
        self.traj_df['timestamp'] = self.traj_df['time'].apply(lambda x: x.timestamp())

        # 获取空间边界，即轨迹数据中的最小和最大经纬度
        spatial_border = traj_df[[X_COL, Y_COL]]
        self.spatial_border = [spatial_border.min().tolist(), spatial_border.max().tolist()]

    def __len__(self):
        # 返回轨迹ID的数量
        return self.traj_ids.shape[0]

    def __getitem__(self, index):
        # 获取指定索引的轨迹
        one_traj = self.traj_df[self.traj_df[TRAJ_ID_COL] == self.traj_ids[index]].copy()
        # 计算时间差（即delta_t）
        one_traj[DT_COL] = one_traj[T_COL] - one_traj[T_COL].iloc[0]
        return one_traj


class PretrainPadder:
    """Collate function for padding pre-training data.
    """

    def __init__(self, device):
        """
        Args:
            device (str): name of the device to put tensors on.
        """
        self.device = device

    def __call__(self, raw_batch):
        """Collat​​e 函数用于将原始轨迹 DataFrames 批次填充为 Tensors。

            参数：
            raw_batch（列表）：每个项目都是一个代表一条轨迹的 `pd.DataFrame`。

            返回：
            torch.FloatTensor：填充后的轨迹特征批次，形状为 (B, L, F)。
            torch.LongTensor：批次中轨迹的有效长度，形状为 (B)。
        """
        traj_batch, valid_lens = [], []
        # 遍历批次中的每个轨迹
        # print('raw_batch:',raw_batch)
        for row in raw_batch:
            traj = row[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()  # 选择经纬度、时间戳、delta_t和道路列
            valid_len = traj.shape[0]  # 轨迹的有效长度
            traj_batch.append(traj)
            valid_lens.append(valid_len)
        
        # 将轨迹数据填充为相同长度，并转换为Tensor
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)

        return traj_batch, valid_lens


class DpPadder:
    """Collate function for padding destination prediction (DP) task data.
    """

    def __init__(self, device, pred_len, pred_cols):
        """
        Args:
            device (str): name of the device to put tensors on.
            pred_len (int): the length of the tail sub-trajectory to remove from the input trajectory.要从输入轨迹中移除的尾部子轨迹的长度。
            pred_cols (list): the columns to predict.
        """
        self.device = device
        self.pred_len = pred_len
        self.pred_cols = pred_cols

    def __call__(self, raw_batch):
        """
        Returns:
            torch.FloatTensor: the padded batch of trajectory features, with shape (B, L, F).填充的轨迹特征批次，形状为 (B, L, F)。
            torch.LongTensor: the valid lengths of trajectories in the batch, with shape (B).批次中轨迹的有效长度，形状为 (B)。
            torch.FloatTensor: the ground truth of the DP task, i.e., features of the last trajectory point, 
            with shape (B, F).即最后一个轨迹点的特征，形状为 (B, F)。
        """
        traj_batch, valid_lens, label_batch = [], [], []
        # 遍历批次中的每个轨迹
        for row in raw_batch:
            traj = row[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()
            traj = traj[:-self.pred_len]  # 去除最后预测长度的部分
            valid_len = traj.shape[0]  # 计算有效长度
            traj_batch.append(traj)
            valid_lens.append(valid_len)

            # 提取预测目标列的数据（例如最后一个轨迹点的特征）
            label = row.iloc[-1][self.pred_cols].to_numpy()
            label_batch.append(label)

        # 将轨迹数据填充为相同长度，并转换为Tensor
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)
        label_batch = torch.from_numpy(np.stack(label_batch, 0).astype(float)).float().to(self.device)

        return traj_batch, valid_lens, label_batch


class TtePadder:
    """Collate function for padding travel time estimation (TTE) task data.
    """

    def __init__(self, device):
        """
        Args:
            device (str): name of the device to put tensors on.
        """
        print('TtePadder')
        self.device = device
        # self.pred_len = pred_len  # 保存pred_len参数
        # self.pred_cols = pred_cols  # 保存pred_cols参数

    def __call__(self, raw_batch):
        """
        Returns:
            torch.FloatTensor: the padded batch of trajectory features, with shape (B, L, F).
            torch.LongTensor: the valid lengths of trajectories in the batch, with shape (B).
            torch.FloatTensor: the ground truth of the TTE task, i.e., travel time of trajectories in minutes, 
            with shape (B).
        """
        traj_batch, valid_lens, label_batch = [], [], []
        # 遍历批次中的每个轨迹
        for row in raw_batch:
            traj = row[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()
            traj[1:, COL_I['temporal']] = -1  # 将轨迹中的时间特征填充为-1
            valid_len = traj.shape[0]  # 计算有效长度
            traj_batch.append(traj)
            valid_lens.append(valid_len)
            # 添加目标：时间估算任务的目标是轨迹的行驶时间（以分钟为单位）
            label_batch.append(row.iloc[-1][DT_COL] / 60)
        
        # 将轨迹数据填充为相同长度，并转换为Tensor
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)
        label_batch = torch.tensor(label_batch).float().to(self.device)

        return traj_batch, valid_lens, label_batch

def fetch_task_padder(padder_name, device, padder_params):
    if padder_name == 'dp':
        task_padder = DpPadder(device, **padder_params)
    elif padder_name == 'tte':#tte任务没有pred_cols,pred_len
        task_padder = TtePadder(device, **padder_params)
    elif padder_name == 'search':  # 对于 search 任务，不需要 padder
        task_padder = None  # 或者可以直接返回 None
    else:
        raise NotImplementedError(f'No Padder named {padder_name}')
    return task_padder



def pad_batch(batch):
    """
    将批次中的所有数组填充到最大长度。

    该函数接受一个包含多个数组的列表，每个数组的形状为 (L, F)，其中 L 是序列长度，F 是特征数量。
    函数会将批次中所有数组填充到最长数组的长度，填充部分使用预定义的值（FEATURE_PAD）。

    参数:
        batch (list): 包含多个数组的列表，数组形状为 (L, F)，L 为序列长度，F 为特征数量。batch (list): the batch of arrays to pad, [(L1, F), (L2, F), ...].

    返回:
        np.array: 一个形状为 (batch_size, max_len, F) 的 NumPy 数组，其中 `batch_size` 是批次中数组的个数，
                  `max_len` 是批次中最长数组的长度，填充部分使用值 `FEATURE_PAD`。
    """
    # 获取批次中数组的最大长度
    max_len = max([arr.shape[0] for arr in batch])

    # 创建一个新的数组，形状为 (len(batch), max_len, F)，并用填充值 (FEATURE_PAD) 填充
    padded_batch = np.full((len(batch), max_len, batch[0].shape[-1]), FEATURE_PAD, dtype=float)

    # 将每个数组的元素复制到填充后的数组中，较短的数组会在末尾填充
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr  # 将原数组的元素复制到对应位置

    return padded_batch



class TrajectorySearchTestdata:
    def __init__(self, test_dataset: TrajClipDataset, spatial_border, num_target=1000, num_negative=5000, neg_random_choice=False):
        """
        初始化 TrajectorySearchTestdata 类，构建用于轨迹搜索的测试数据。

        参数:
            test_dataset (TrajClipDataset): 用于测试的轨迹数据集。
            spatial_border (tuple): 空间边界，用于标准化轨迹数据。
            num_target (int, optional): 要选取的目标轨迹数，默认为 1000。
            num_negative (int, optional): 每个查询轨迹对应的负样本数，默认为 5000。
            neg_random_choice (bool, optional): 是否随机选择负样本，默认为 False。
        """
        trajs = []
        # 遍历测试数据集中的所有轨迹 ID
        for traj_id in tqdm(test_dataset.traj_ids, desc='Gathering trips', total=len(test_dataset.traj_ids), leave=False, ncols=70):
            one_traj = test_dataset.traj_df[test_dataset.traj_df[TRAJ_ID_COL] == traj_id].copy()
            # 计算时间差 (DT)
            one_traj[DT_COL] = one_traj[T_COL] - one_traj[T_COL].iloc[0]
            # 只保留特定的列，转换为 numpy 数组并存储
            traj = one_traj[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL]].to_numpy()
            trajs.append(traj)
        self.trajs = np.array(trajs, dtype=object)

        # 选择目标轨迹的数量，保证不超过轨迹总数减一
        num_target = min(len(self.trajs) - 1, num_target)
        random.seed(10)
        sampled_trip_ids = random.sample(range(len(self.trajs)), num_target)
        
        # 从选中的轨迹中生成查询轨迹和目标轨迹
        qry_trips = [t[::2] for t in self.trajs[sampled_trip_ids]]  # 查询轨迹：取偶数索引
        tgt_trips = [t[1::2] for t in self.trajs[sampled_trip_ids]]  # 目标轨迹：取奇数索引
        self.hopqrytgt = np.array(qry_trips + tgt_trips, dtype=object)

        # 获取所有轨迹的目标轨迹
        all_hoptgts = [t[1::2] for t in self.trajs]
        self.all_hoptgts = np.array(all_hoptgts, dtype=object)

        # 选择负样本的数量
        num_negative = min(len(self.trajs) - num_target, num_negative)
        
        if neg_random_choice:
            # 如果选择随机负样本
            neg_indices = []
            for i in trange(num_target, desc='Gathering sim idx'):
                # 从所有轨迹中选择负样本
                neg_trip_ids = np.delete(np.arange(len(self.trajs)), sampled_trip_ids[i])
                neg_indice = np.random.choice(neg_trip_ids, num_negative, replace=False)
                neg_indices.append(neg_indice)
        else:
            # 如果选择空间边界方法
            select_index = COL_I['spatial']
            spatial_border = np.array(spatial_border)
            kseg_trips = []
            for arr in tqdm(self.trajs, desc='Gathering kseg trips', total=len(self.trajs)):
                # 将轨迹按空间边界标准化
                norm_spatial_arr = (arr[..., select_index] - spatial_border[0]) / (spatial_border[1] - spatial_border[0])
                kseg_arr = self.resample_to_k_segments(norm_spatial_arr, MIN_TRIP_LEN)  # 重采样轨迹
                kseg_trips.append(kseg_arr)
            kseg_trips = np.stack(kseg_trips)

            # 进行最远邻居搜索
            qry_trips = self.hopqrytgt[:num_target]
            neg_euclidean = lambda x, y: -euclidean_distances(x.reshape(1, -1), y.reshape(1, -1))
            farthest_knn = NearestNeighbors(n_neighbors=len(kseg_trips) - 10, metric=neg_euclidean)
            farthest_knn.fit(kseg_trips)

            qry_indices = np.arange(num_target)
            neg_indices = []
            for arr in tqdm(kseg_trips[sampled_trip_ids], desc='Gathering sim idx', total=num_target):
                # 选择最远邻居作为负样本
                farthest_idx = farthest_knn.kneighbors([arr], return_distance=False)
                # 从最远邻居中随机选择负样本
                farthest_idx = np.random.choice(farthest_idx[0], num_negative, replace=False)
                neg_indices.append(farthest_idx)
        self.neg_indices = np.array(neg_indices)
        print("neg_indices shape: ", self.neg_indices.shape)

        # 保存文件名的构建
        self.hopqrytgt_savename = f"hopqrytgt-{num_target}"
        self.neg_indices_savename = f"hopnearernegindex-{num_target}-{num_negative}-v2" if not neg_random_choice else f"hoprandomnegindex-{num_target}-{num_negative}"

    def save_search_meta(self, meta_dir):
        """
        保存搜索元数据到指定目录。

        参数:
            meta_dir (str): 保存元数据的目录路径。
        """
        utils.create_if_noexists(meta_dir)
        np.save(os.path.join(meta_dir, "all_hoptgts.npy"), self.all_hoptgts)
        np.save(os.path.join(meta_dir, f"{self.hopqrytgt_savename}.npy"), self.hopqrytgt)
        np.save(os.path.join(meta_dir, f"{self.neg_indices_savename}.npy"), self.neg_indices)
        print("Saved meta to", meta_dir)

    def get_search_meta(self):
        """
        获取搜索的元数据。

        返回:
            tuple: 返回包含 all_hoptgts, hopqrytgt, 和 neg_indices 的元组。
        """
        return self.all_hoptgts, self.hopqrytgt, self.neg_indices
    
    @staticmethod
    def parse_label(length):
        """
        根据轨迹长度，划分查询轨迹和目标轨迹的索引。

        参数:
            length (int): 轨迹的总长度。

        返回:
            tuple: 返回查询轨迹和目标轨迹的索引列表。
        """
        qry_idx = list(range(int(length / 2)))
        tgt_idx = list(range(int(length / 2), length))
        return qry_idx, tgt_idx

    @staticmethod
    def cal_pres_and_labels(query, target, negs):
        """
        计算查询和目标轨迹的距离，以及负样本的标签。

        参数:
            query (ndarray): 查询轨迹，形状为 (N, d)。
            target (ndarray): 目标轨迹，形状为 (N, d)。
            negs (ndarray): 负样本轨迹，形状为 (N, n, d)。

        返回:
            tuple: 返回计算出的精度 (pres) 和标签 (labels)。
        """
        num_queries = query.shape[0]
        num_targets = target.shape[0]
        num_negs = negs.shape[1]
        print("query: ", query.shape)
        print("target: ", target.shape)
        print("neg: ", negs.shape)
        assert num_queries == num_targets, "Number of queries and targets should be the same."

        query_t = repeat(query, 'nq d -> nq nt d', nt=num_targets)
        query_n = repeat(query, 'nq d -> nq nn d', nn=num_negs)
        target = repeat(target, 'nt d -> nq nt d', nq=num_queries)
        # negs = repeat(negs, 'nn d -> nq nn d', nq=num_queries)

        dist_mat_qt = np.linalg.norm(query_t - target, ord=2, axis=2)
        dist_mat_qn = np.linalg.norm(query_n - negs, ord=2, axis=2)
        dist_mat = np.concatenate([dist_mat_qt[np.eye(num_queries).astype(bool)][:, None], dist_mat_qn], axis=1)

        pres = -1 * dist_mat

        labels = np.zeros(num_queries)

        return pres, labels
    
    @staticmethod
    def resample_to_k_segments(trip, kseg):
        """
        Resample a trajectory to k segments.将轨迹重采样成 k 个段。
        :return: a numpy array of shape (kseg * 3,)
        ndarray: 重采样后的轨迹，形状为 (kseg * 3,)，即每段的平均值。
        trip (ndarray): 原始轨迹。
        kseg (int): 目标段数。
        """
        ksegs = []
        seg = len(trip) // kseg

        for i in range(kseg):
            if i == kseg - 1:
                ksegs.append(np.mean(trip[i * seg:], axis=0))
            else:
                ksegs.append(np.mean(trip[i * seg: i * seg + seg], axis=0))
        ksegs = np.array(ksegs).reshape(-1)

        return ksegs

def load_trajSearch_testdata(search_meta_dir, num_target=1000, num_negative=5000, neg_random_choice=False):
    """
    加载轨迹搜索的测试数据，包括目标轨迹、查询轨迹和负样本轨迹。

    参数:
        search_meta_dir (str): 存储元数据的目录路径。
        num_target (int, optional): 要加载的目标轨迹数量，默认为 1000。
        num_negative (int, optional): 每个查询轨迹对应的负样本数量，默认为 5000。
        neg_random_choice (bool, optional): 是否选择随机负样本，默认为 False。

    返回:
        tuple: 返回包含 alltrajtgt, hopqrytgt, 和 neg_indices 的元组。
    """
    # 根据是否选择随机负样本，设置负样本的文件名
    neg_indices_metaname = f"hopnearernegindex-{num_target}-{num_negative}-v2.npy" if not neg_random_choice else \
                             f"hoprandomnegindex-{num_target}-{num_negative}.npy"
    print("neg_indices_type:", neg_indices_metaname)
    
    # 加载元数据文件
    alltrajtgt = np.load(os.path.join(search_meta_dir, "all_hoptgts.npy"), allow_pickle=True)
    hopqrytgt = np.load(os.path.join(search_meta_dir, f"hopqrytgt-{num_target}.npy"), allow_pickle=True)
    neg_indices = np.load(os.path.join(search_meta_dir, neg_indices_metaname), allow_pickle=True)

    # 返回加载的数据
    return alltrajtgt, hopqrytgt, neg_indices


class TrajectorySearchDataset(Dataset):
    """
    用于轨迹搜索的数据集类
    """
    def __init__(self, trajs):
        super().__init__()
        self.trajs = trajs

    def __len__(self):
        """
        返回数据集中的轨迹数目。

        返回:
            int: 数据集中轨迹的数量。
        """
        return self.trajs.shape[0]

    def __getitem__(self, index):
        """
        获取指定索引的轨迹数据。

        参数:
            index (int): 要获取的轨迹索引。

        返回:
            ndarray: 对应轨迹的数组表示。
        """
        one_traj = self.trajs[index].copy()
        return one_traj

class SearchPadder:
    """
    用于轨迹搜索的填充函数类，负责将批次数据填充到相同的长度。
    """

    def __init__(self, device):
        """
        初始化填充器，设置设备。

        参数:
            device (str): 将张量放置在哪个设备上（例如 "cuda" 或 "cpu"）。
        """
        self.device = device

    def __call__(self, raw_batch):
        """
        将原始批次数据填充并转换为张量。

        参数:
            raw_batch (list): 每个项都是一个表示单个轨迹的 `pd.DataFrame`。

        返回:
            tuple: 返回两个张量，第一个是填充后的轨迹数据，第二个是有效轨迹的长度。
                - torch.FloatTensor: 填充后的轨迹数据，形状为 (B, L, F)。
                - torch.LongTensor: 每个轨迹的有效长度，形状为 (B)。
        """
        traj_batch, valid_lens = [], []
        for traj in raw_batch:
            valid_len = traj.shape[0]
            traj_batch.append(traj)
            valid_lens.append(valid_len)
        
        # 将轨迹数据填充到相同长度
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)

        return traj_batch, valid_lens



# 程序入口
if __name__ == '__main__':
    import json
    from argparse import ArgumentParser
    import os
    import pandas as pd

    print('执行data_main')

    parser = ArgumentParser()
    
    # 添加命令行参数：设置文件名称
    # -s 或 --settings：指定设置文件的名称
    # help：参数的帮助信息
    # type：参数的类型（本例中为字符串）
    # default：参数的默认值（本例中为 "local_test_search"）
    parser.add_argument('-s', '--settings', help='name of the settings file to use', type=str, default="local_test_search") 

    # 解析命令行参数
    args = parser.parse_args()

    # 加载设置文件
    # os.path.join：连接目录和文件名，确保正确的路径分隔符
    # open：打开文件，'r' 表示以只读方式打开
    # json.load：从文件中加载 JSON 数据
    with open(os.path.join('settings', f'{args.settings}.json'), 'r') as fp:
        # settings：加载的 JSON 数据，应该是一个列表或字典
        settings = json.load(fp)

    # 遍历设置文件中的多个设置
    for setting_i, setting in enumerate(settings):
        # 打印当前设置的索引和总数
        print(f'===SETTING {setting_i}/{len(settings)}===')

        # 获取设置的保存名称，如果没有指定，则为 None
        SAVE_NAME = setting.get('save_name', None)

        # 检查设置中是否包含 'test' 键
        if 'test' in setting:
            # 加载训练和测试数据集
            # pd.read_hdf：从 HDF5 文件中读取数据
            train_traj_df = pd.read_hdf(setting['dataset']['train_traj_df'], key='trips')
            test_traj_df = pd.read_hdf(setting['dataset']['test_traj_df'], key='trips')

            # 创建训练和测试数据集对象
            train_dataset = TrajClipDataset(traj_df=train_traj_df)
            test_dataset = TrajClipDataset(traj_df=test_traj_df)

            # 创建搜索测试数据对象
            # TrajectorySearchTestdata：一个类，用于创建搜索测试数据
            # test_dataset：测试数据集
            # spatial_border：训练数据集的空间边界
            # **setting['test']["search_data_params"]：搜索数据参数，作为关键字参数传递
            simTrajSearch_testData = TrajectorySearchTestdata(test_dataset, spatial_border=train_dataset.spatial_border, **setting['test']["search_data_params"])

            # 获取评估数据集的名称
            eval_dataset = os.path.basename(setting['dataset']['test_traj_df']).split(".")[0]

            # 创建元数据目录
            meta_dir = os.path.join(SEARCH_META_DIR, eval_dataset)

            # 保存搜索元数据
            # simTrajSearch_testData.save_search_meta：保存搜索元数据的方法
            simTrajSearch_testData.save_search_meta(meta_dir)