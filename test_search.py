import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载数据
predictions = np.load('/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/predictions/chengdu.h5/local_test_search/search/predictions.npy')  # 预测的轨迹
targets = np.load('/home/jsj201-11/mount1/xzz/基线/PTrajM-C973_meta/predictions/chengdu.h5/local_test_search/search/targets.npy')  # 真实的轨迹
print(f"Predictions shape: {predictions.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Example prediction: {predictions[0]}")
print(f"Example target: {targets[0]}")

# 计算 Acc@1 和 Acc@5
def calculate_accuracy_at_n(predictions, targets, n=5):
    acc_at_n = 0
    total = len(predictions)
    
    for i in range(total):
        pred = predictions[i]
        target = targets[i]
        
        # 计算余弦相似度
        sim_scores = cosine_similarity([pred], predictions)[0]
        
        # 排序并获取前N个预测
        sorted_indices = np.argsort(sim_scores)[::-1]  # 降序排序
        top_n_predictions = sorted_indices[:n]
        
        # 判断Acc@1，是否预测结果在前1个
        if target in top_n_predictions[:1]:
            acc_at_n += 1
        # 判断Acc@5，是否预测结果在前5个
        elif target in top_n_predictions:
            acc_at_n += 1

    return acc_at_n / total

# 计算 Mean Rank
def calculate_mean_rank(predictions, targets):
    ranks = []
    for i in range(len(predictions)):
        pred = predictions[i]
        target = targets[i]
        
        # 计算余弦相似度
        sim_scores = cosine_similarity([pred], predictions)[0]
        
        # 获取目标轨迹的排名
        sorted_indices = np.argsort(sim_scores)[::-1]  # 降序排序
        rank = np.where(sorted_indices == target)[0][0] + 1  # 加1是因为排名从1开始
        ranks.append(rank)

    return np.mean(ranks)


# 计算评价指标
acc_at_1 = calculate_accuracy_at_n(predictions, targets, n=1)
acc_at_5 = calculate_accuracy_at_n(predictions, targets, n=5)
mean_rank = calculate_mean_rank(predictions, targets)

print(f"Acc@1: {acc_at_1:.4f}")
print(f"Acc@5: {acc_at_5:.4f}")
print(f"Mean Rank: {mean_rank:.4f}")
