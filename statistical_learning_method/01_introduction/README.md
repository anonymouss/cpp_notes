# Basics

TP: 正类预测为正 （True Positive，真正）

FN: 正类预测为负 （False Negative，假负）

FP: 负类预测为正 （False Positive，假正）

TN: 负类预测为负 （True Negative，真负）

准确率：accuracy = (TP + TN) / (TP + FN + FP + TN)      # 预测结果正确的概率

错误率：error = (FP + FN) / (TP + FN + FP + TN)         # 预测结果错误的概率

灵敏度：sensitive = (TP) / (TP + FN)                    # 正样本被分类正确的概率

特效度：specificity = (TN) / (TN + FP)                  # 负样本被分类正确的概率

精度：  precision = (TP) / (TP + FP)                    # 分类为正的样本确实为正的概率

召回率：recall = (TP) / (TP + FN) = 灵敏度

F1值：  F1 = (precision + recall) / 2 = (2TP) / (2TP + FP + FN)
