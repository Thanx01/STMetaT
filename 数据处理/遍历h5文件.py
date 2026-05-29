import h5py

# 打开 .h5 文件
file_path = '/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/datasets_from_Lin/chengdu/chengdu_train.h5'
with h5py.File(file_path, 'r') as f:
    # 打印文件的所有键（即数据集的名称）
    print("文件包含的所有键:")
    for key in f.keys():
        print(key)

    group_name = 'network_info'
    if group_name in f:
        group = f[group_name]
        print(f"\n组 '{group_name}' 的成员:")
        for sub_key in group.keys():
            print(f"  子成员: {sub_key}")
            sub_item = group[sub_key]
            if isinstance(sub_item, h5py.Dataset):  # 如果是数据集
                print(f"    类型: 数据集")
                print(f"    数据集形状: {sub_item.shape}")
                print(f"    数据集类型: {sub_item.dtype}")
                print(f"    数据（前5个元素）: {sub_item[:5]}")  # 打印前5个元素
            elif isinstance(sub_item, h5py.Group):  # 如果是子组
                print(f"    类型: 子组")
                print(f"    子组成员: {list(sub_item.keys())}")  # 查看子组内的成员
            else:
                print(f"    其他类型: {type(sub_item)}")
    else:
        print(f"\n未找到名为 '{group_name}' 的组。")
        

    # 查看 'pois' 组的内容
    group_name = 'pois'
    if group_name in f:
        group = f[group_name]
        print(f"\n组 '{group_name}' 的成员:")
        for sub_key in group.keys():
            print(f"  子成员: {sub_key}")
            sub_item = group[sub_key]
            if isinstance(sub_item, h5py.Dataset):  # 如果是数据集
                print(f"    类型: 数据集")
                print(f"    数据集形状: {sub_item.shape}")
                print(f"    数据集类型: {sub_item.dtype}")
                print(f"    数据（前5个元素）: {sub_item[:5]}")  # 打印前5个元素
            elif isinstance(sub_item, h5py.Group):  # 如果是子组
                print(f"    类型: 子组")
                print(f"    子组成员: {list(sub_item.keys())}")  # 查看子组内的成员
            else:
                print(f"    其他类型: {type(sub_item)}")

    # 查看 'road_info' 组的内容
    group_name = 'road_info'
    if group_name in f:
        group = f[group_name]
        print(f"\n组 '{group_name}' 的成员:")
        for sub_key in group.keys():
            print(f"  子成员: {sub_key}")
            sub_item = group[sub_key]
            if isinstance(sub_item, h5py.Dataset):  # 如果是数据集
                print(f"    类型: 数据集")
                print(f"    数据集形状: {sub_item.shape}")
                print(f"    数据集类型: {sub_item.dtype}")
                print(f"    数据（前5个元素）: {sub_item[:5]}")  # 打印前5个元素
            elif isinstance(sub_item, h5py.Group):  # 如果是子组
                print(f"    类型: 子组")
                print(f"    子组成员: {list(sub_item.keys())}")  # 查看子组内的成员
            else:
                print(f"    其他类型: {type(sub_item)}")


    group_name = 'trip_info'
    if group_name in f:
        group = f[group_name]
        print(f"\n组 '{group_name}' 的成员:")
        for sub_key in group.keys():
            print(f"  子成员: {sub_key}")
            sub_item = group[sub_key]
            if isinstance(sub_item, h5py.Dataset):  # 如果是数据集
                print(f"    类型: 数据集")
                print(f"    数据集形状: {sub_item.shape}")
                print(f"    数据集类型: {sub_item.dtype}")
                print(f"    数据（前5个元素）: {sub_item[:5]}")  # 打印前5个元素
            elif isinstance(sub_item, h5py.Group):  # 如果是子组
                print(f"    类型: 子组")
                print(f"    子组成员: {list(sub_item.keys())}")  # 查看子组内的成员
            else:
                print(f"    其他类型: {type(sub_item)}")
    else:
        print(f"\n未找到名为 '{group_name}' 的组。")
        
        
    group_name = 'trips'
    if group_name in f:
        group = f[group_name]
        print(f"\n组 '{group_name}' 的成员:")
        for sub_key in group.keys():
            print(f"  子成员: {sub_key}")
            sub_item = group[sub_key]
            if isinstance(sub_item, h5py.Dataset):  # 如果是数据集
                print(f"    类型: 数据集")
                print(f"    数据集形状: {sub_item.shape}")
                print(f"    数据集类型: {sub_item.dtype}")
                print(f"    数据（前5个元素）: {sub_item[:5]}")  # 打印前5个元素
            elif isinstance(sub_item, h5py.Group):  # 如果是子组
                print(f"    类型: 子组")
                print(f"    子组成员: {list(sub_item.keys())}")  # 查看子组内的成员
            else:
                print(f"    其他类型: {type(sub_item)}")
    else:
        print(f"\n未找到名为 '{group_name}' 的组。")
        
        
        
        