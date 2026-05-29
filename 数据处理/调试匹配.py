import numpy as np
import h5py

# 加载嵌入向量
embeddings = np.load('/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/samples/small_chengdu_poi_embed.npy')
print(f"嵌入数据的形状: {embeddings.shape}")
print(f"嵌入数据的类型: {embeddings.dtype}")

# 加载 HDF5 文件
with h5py.File('/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/datasets_from_Lin/chengdu/chengdu.h5', 'r') as f:
    # 查看 pois 数据
    pois = f['pois']
    print(pois['axis0'][:])  # 输出 axis0，查看标签名称
    print(pois['axis1'][:])  # 输出 axis1，查看 road ID

    # 确保嵌入与 pois 的匹配
    print("嵌入向量和 pois 是否匹配？")
    print(embeddings.shape[0] == len(pois['axis1']))  # 验证嵌入数是否与道路 ID 数一致
