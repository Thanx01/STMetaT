import numpy as np
import math

def to_radians(deg):
    return deg * math.pi / 180.0

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算地球表面上两点的球面距离（米）
    参数为: 经度1、纬度1、经度2、纬度2
    """
    R = 6371000  # 地球平均半径，单位：米
    d_lon = to_radians(lon2 - lon1)
    d_lat = to_radians(lat2 - lat1)
    a = (math.sin(d_lat / 2)**2 +
         math.cos(to_radians(lat1)) * math.cos(to_radians(lat2)) * math.sin(d_lon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

def evaluate_dp(predictions_file, targets_file):
    # 1) 读入预测和真实坐标
    preds = np.load(predictions_file)   # shape (N, 2) => (lng, lat)
    trues = np.load(targets_file)       # shape (N, 2) => (lng, lat)

    # 检查形状是否相同
    assert preds.shape == trues.shape, \
        f"Shape mismatch: preds.shape={preds.shape}, trues.shape={trues.shape}"

    N = preds.shape[0]
    distances = []
    
    # 2) 遍历每个样本，计算球面距离
    for i in range(N):
        lon_pred, lat_pred = preds[i]  # (lng, lat)
        lon_true, lat_true = trues[i]  
        dist = haversine_distance(lon_pred, lat_pred, lon_true, lat_true)
        distances.append(dist)
    
    distances = np.array(distances)

    # 3) 计算MAE, RMSE
    mae_m = np.mean(np.abs(distances))            # 平均绝对误差（米）
    rmse_m = np.sqrt(np.mean(distances**2))       # 均方根误差（米）

    return mae_m, rmse_m


if __name__ == "__main__":
    # 假设你的文件在 ./predictions/local_test/dp/
    # pred_file = "./predictions/meta/chengdu.h5/local_test/dp/predictions.npy"
    # true_file = "./predictions/meta/chengdu.h5/local_test/dp/targets.npy"
    pred_file = "/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/predictions/meta/chengdu.h5/local_test/dp/chengdu.h5_tw2700s_grid0.014_bs16_predictions.npy"
    true_file = "/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/predictions/meta/chengdu.h5/local_test/dp/chengdu.h5_tw2700s_grid0.014_bs16_targets.npy"

    mae_m, rmse_m = evaluate_dp(pred_file, true_file)
    
    print(f"DP Evaluation Results:")
    print(f"MAE (m): {mae_m:.4f}")
    print(f"RMSE (m): {rmse_m:.4f}")


#tte
# import numpy as np
# import torch
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import math
# import json

# # 加载预测和目标值
# predictions = np.load("/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/predictions/meta/chengdu.h5/local_test/tte/chengdu.h5_tw2700s_grid0.1_bs16_predictions.npy")
# targets = np.load("/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/predictions/meta/chengdu.h5/local_test/tte/chengdu.h5_tw2700s_grid0.1_bs16_targets.npy")

# # 打印预测和目标的形状
# print(f"Predictions shape: {predictions.shape}")
# print(f"Targets shape: {targets.shape}")

# # 如果 predictions 维度是 (batch_size, 2)，选择一个维度
# predictions = predictions[:, 0]  # 选择第一个维度作为预测值

# # targets 已经是一维数组，不需要进一步处理
# # targets = targets[:, 0]  # 不需要这行

# # 计算 MAE（平均绝对误差）
# mae = mean_absolute_error(targets, predictions) * 100.0

# # 计算 RMSE（均方根误差）
# rmse = math.sqrt(mean_squared_error(targets, predictions)) * 100.0
 
# # 计算 MAPE（平均绝对百分比误差）
# mape = np.mean(np.abs((targets - predictions) / targets)) * 100.0

# # 输出结果
# print(f"MAE: {mae:.4f} seconds")
# print(f"RMSE: {rmse:.4f} seconds")
# print(f"MAPE: {mape:.4f} %")

# # 将结果转换为标准的 Python 数值类型
# results = {
#     "MAE": float(mae),
#     "RMSE": float(rmse),
#     "MAPE": float(mape)
# }

# # 保存到一个文件中
# with open('evaluation_metrics.json', 'w') as f:
#     json.dump(results, f, indent=4)


# trajfm
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import json

# # 定义你的数据路径
# pred_path = "/home/jsj201-11/mount1/xzz/基线/TrajFM/datasets_from_Lin/xian/xian.h5_tp_predictions.npz"
# target_path = "/home/jsj201-11/mount1/xzz/基线/TrajFM/datasets_from_Lin/xian/xian.h5_tp_targets.npz"

# # 加载数据 (npz文件加载方式)
# predictions_npz = np.load(pred_path)
# targets_npz = np.load(target_path)

# # 检查npz内部数据
# predictions = predictions_npz['arr_0']
# targets = targets_npz['arr_0']

# # 打印预测和目标的形状
# print(f"Original predictions shape: {predictions.shape}")
# print(f"Original targets shape: {targets.shape}")

# # 如果predictions和targets为三维(batch, seq_len, coord)，例如(batch, seq_len, 2)
# # 这里假设目标和预测维度都是 (样本数, 序列长度, 2)
# # 若你只评估位置的整体误差，可以使用所有点直接展平计算
# if predictions.ndim == 3:
#     predictions = predictions.reshape(-1, predictions.shape[-1])
# if targets.ndim == 3:
#     targets = targets.reshape(-1, targets.shape[-1])

# # 确保 predictions 和 targets 维度一致
# assert predictions.shape == targets.shape, "预测值和目标值维度不匹配，请检查数据！"

# # 计算每个坐标维度的MAE和RMSE，再计算整体误差
# mae_x = mean_absolute_error(targets[:, 0], predictions[:, 0]) * 10.0
# mae_y = mean_absolute_error(targets[:, 1], predictions[:, 1]) * 10.0
# rmse_x = np.sqrt(mean_squared_error(targets[:, 0], predictions[:, 0])) * 10.0
# rmse_y = np.sqrt(mean_squared_error(targets[:, 1], predictions[:, 1])) * 10.0

# # 整体误差（对x和y取平均）
# mae = (mae_x + mae_y) / 2 * 1000.0
# rmse = (rmse_x + rmse_y) / 2 * 1000.0

# # 计算MAPE (如果目标坐标中有0值，避免除以0)
# non_zero_targets = np.where(targets == 0, 1e-6, targets)
# mape = np.mean(np.abs((targets - predictions))) * 1000.0

# # 打印结果
# print(f"MAE: {mae:.4f}")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAPE: {mape:.4f}%")

# # 保存结果为json
# results = {
#     "MAE": float(mae),
#     "RMSE": float(rmse),
#     "MAPE": float(mape)
# }

# with open("evaluation_results.json", "w") as f:
#     json.dump(results, f, indent=4)

# print("评估结果已保存到 evaluation_results.json")






# import numpy as np
# import torch
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import math
# import json

# # 定义你的数据路径
# pred_path = "/home/jsj201-11/mount1/xzz/基线/TrajFM/datasets_from_Lin/chengdu/chengdu.h5_tp_predictions.npz"
# target_path = "/home/jsj201-11/mount1/xzz/基线/TrajFM/datasets_from_Lin/chengdu/chengdu.h5_tp_targets.npz"

# # 加载数据 (npz文件加载方式)
# predictions_npz = np.load(pred_path)
# targets_npz = np.load(target_path)

# # 检查npz内部数据
# predictions = predictions_npz['arr_0']
# targets = targets_npz['arr_0']

# # 打印预测和目标的形状
# print(f"Original predictions shape: {predictions.shape}")
# print(f"Original targets shape: {targets.shape}")

# # 如果 predictions 和 targets 为三维 (batch, seq_len, coord)，例如 (batch, seq_len, 2)
# # 展平为 (总点数, 2) 或 (总点数,)
# if predictions.ndim == 3:
#     predictions = predictions.reshape(-1, predictions.shape[-1])
# if targets.ndim == 3:
#     targets = targets.reshape(-1, targets.shape[-1])

# # 如果 predictions 是 (总点数, 2)，选择时间维度（假设第 0 维是时间）
# if predictions.shape[-1] == 2:
#     predictions = predictions[:, 0]  # 提取时间预测
# else:
#     raise ValueError("predictions 最后一维应为 2，但当前为 {}".format(predictions.shape[-1]))

# # 如果 targets 是 (总点数, 2)，也提取时间维度；如果是一维，则保持不变
# if targets.ndim == 2 and targets.shape[-1] == 2:
#     targets = targets[:, 0]  # 提取时间目标
# elif targets.ndim == 1:
#     pass  # 已经是一维，无需处理
# else:
#     raise ValueError("targets 维度不正确，当前为 {}".format(targets.shape))

# # 确保 predictions 和 targets 维度一致
# assert predictions.shape == targets.shape, "预测值和目标值维度不匹配，请检查数据！"

# # 计算 MAE（平均绝对误差）
# mae = mean_absolute_error(targets, predictions) * 10000.0

# # 计算 RMSE（均方根误差）
# rmse = math.sqrt(mean_squared_error(targets, predictions)) * 10000.0

# # 计算 MAPE（平均绝对百分比误差），避免除以零
# mask = targets != 0  # 创建非零掩码
# mape = np.mean(np.abs(targets[mask] - predictions[mask])) * 1000.0

# # 打印结果
# print(f"MAE: {mae:.4f} seconds")
# print(f"RMSE: {rmse:.4f} seconds")
# print(f"MAPE: {mape:.4f} %")

# # 保存结果为json
# results = {
#     "MAE": float(mae),
#     "RMSE": float(rmse),
#     "MAPE": float(mape)
# }

# with open("evaluation_results.json", "w") as f:
#     json.dump(results, f, indent=4)

# print("评估结果已保存到 evaluation_results.json")