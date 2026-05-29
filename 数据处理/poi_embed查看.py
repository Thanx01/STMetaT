import numpy as np

# 文件路径
file_path = '/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/samples/small_chengdu_poi_embed.npy'

# 加载 .npy 文件
embedding_data = np.load(file_path)

# 查看数据的形状和类型
print(f"Data shape: {embedding_data.shape}")
print(f"Data type: {embedding_data.dtype}")

# 查看前几个元素（可以帮助你了解数据的大致内容）
print(f"First few entries:\n{embedding_data[:5]}")  # 查看前五个元素
