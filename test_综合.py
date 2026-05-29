import numpy as np
import utils  # 确保 utils.py 在同一目录下或在 PYTHONPATH 中

def evaluate_dp(predictions_file, targets_file):
    preds = np.load(predictions_file)  # shape (N, 2) => (lng, lat)
    trues = np.load(targets_file)      # shape (N, 2) => (lng, lat)
    
    assert preds.shape == trues.shape, \
        f"Shape mismatch: preds.shape={preds.shape}, trues.shape={trues.shape}"
    
    lng_col = 0
    lat_col = 1
    
    metric = utils.cal_distance_metric(trues, preds, lng_col, lat_col)
    
    mae_m = metric['distance_mae']
    rmse_m = metric['distance_rmse']
    
    return mae_m, rmse_m

# def evaluate_tte(predictions_file, targets_file):
#     preds = np.load(predictions_file)  # shape (N,) 或 (N, 1)
#     trues = np.load(targets_file)      # shape (N,)
    
#     if preds.ndim == 2:
#         preds = preds[:, 0]
    
#     assert preds.shape == trues.shape, \
#         f"Shape mismatch: preds.shape={preds.shape}, trues.shape={trues.shape}"
    
#     metric = utils.cal_regression_metric(trues, preds)
    
#     mae = metric['mae'] * 100.0
#     rmse = metric['rmse'] * 100.0
#     mape = metric['mape'] * 100.0
    
#     return mae, rmse, mape


def evaluate_tte(predictions_file, targets_file):
    # 加载预测值和真实值
    preds = np.load(predictions_file)  # 形状应为 (N,) 或 (N, 1)
    trues = np.load(targets_file)      # 形状为 (N, 2)

    # 提取到达时间（假设在第 0 列）
    trues = trues[:, 0]  # 变成形状 (N,)

    # 如果 preds 是二维数组，也提取第一列
    if preds.ndim == 2:
        preds = preds[:, 0]

    # 确保形状匹配
    assert preds.shape == trues.shape, \
        f"Shape mismatch: preds.shape={preds.shape}, trues.shape={trues.shape}"

    # 计算回归指标
    metric = utils.cal_regression_metric(trues, preds)

    # 提取 MAE, RMSE, MAPE 并转换为百分比
    mae = metric['mae'] * 100.0
    rmse = metric['rmse'] * 100.0
    mape = metric['mape'] * 100.0

    return mae, rmse, mape

def evaluate_search(predictions_file, targets_file):
    preds = np.load(predictions_file)  # shape (N, num_class)
    trues = np.load(targets_file)      # shape (N,)
    
    assert preds.shape[0] == trues.shape[0], \
        f"Shape mismatch: preds.shape={preds.shape}, trues.shape={trues.shape}"
    
    metric = utils.cal_search_metric(trues, preds)
    
    acc1 = metric['acc@1']
    acc5 = metric['acc@5']
    mean_rank = metric['mean_rank']
    
    return acc1, acc5, mean_rank

if __name__ == "__main__":
    task = "dp"  # 可选 "dp"、"tte" 或 "search"
    pred_file = "/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/predictions/local_test/dp_predictions.npy"
    true_file = "/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/predictions/local_test/dp_targets.npy"
    
    if task == "dp":
        mae_m, rmse_m = evaluate_dp(pred_file, true_file)
        print(f"DP Evaluation Results:")
        print(f"MAE (m): {mae_m:.4f}")
        print(f"RMSE (m): {rmse_m:.4f}")
    elif task == "tte":
        pred_file = "/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/predictions/chengdu.h5/local_test/dp/predictions.npy"
        true_file = "/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/predictions/chengdu.h5/local_test/dp/targets.npy"
        mae, rmse, mape = evaluate_tte(pred_file, true_file)
        print(f"TTE Evaluation Results:")
        print(f"MAE: {mae:.4f} seconds")
        print(f"RMSE: {rmse:.4f} seconds")
        print(f"MAPE: {mape:.4f} %")
    elif task == "search":
        acc1, acc5, mean_rank = evaluate_search(pred_file, true_file)
        print(f"Search Evaluation Results:")
        print(f"Acc@1: {acc1:.4f}")
        print(f"Acc@5: {acc5:.4f}")
        print(f"Mean Rank: {mean_rank:.4f}")
    else:
        raise ValueError(f"Unknown task: {task}")