import numpy as np

def check_embedding_file(file_path):
    # 加载文件
    try:
        embedding_data = np.load(file_path)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return
    
    # 打印嵌入数据的基本信息
    print(f"嵌入数据文件: {file_path}")
    print(f"嵌入数据的形状: {embedding_data.shape}")
    print(f"嵌入数据的类型: {embedding_data.dtype}")
    
    # 打印前几个嵌入向量，查看样本数据
    print("前几个嵌入向量：")
    print(embedding_data[:5])

# 路径指向你的 road_embed.npy 文件
file_path = '/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/samples/small_chengdu_road_embed.npy'

# 检查 road_embed.npy 文件
check_embedding_file(file_path)
