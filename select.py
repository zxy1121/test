import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

data = pd.read_excel('C:\\Users\\hp\\Desktop\\pythonProject1\\after_process_Molecular_Descriptor.xlsx', header=0, index_col=0)

feature = data.iloc[:, :729]
target = data.iloc[:, 730]
train_features, test_features, train_target, test_target = train_test_split(feature, target, test_size=0.25, random_state=0)
# 创建随机森林分类器
randomforest = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1, oob_score=False)
# 训练模型
model = randomforest.fit(train_features, train_target)
test_predict = randomforest.predict(test_features)


mse = metrics.mean_absolute_error(test_target, test_predict)
ssr = ((test_predict - test_target.mean()) ** 2).sum()
sst = ((test_target - test_target.mean()) ** 2).sum()
r2 = ssr / sst
print(mse)
print(r2)                  # 得到模型的mse评价指标与r-square评价指标

# 计算特征的重要性
importances = model.feature_importances_
# 将特征的重要性排序
indices = np.argsort(importances)[::-1]
indices_20 = indices[0:20]
# 按照特征的重要性对特征名重新排序
names = [data.columns[i] for i in indices]
names_20 = names[0:20]
# 可视化 重要性排序结果
plt.figure()
plt.title("Feature Importance")
# 添加数据条
plt.bar(range(20), importances[indices_20])
# 将下标作为x轴标签
plt.xticks(range(20), names_20, rotation=90)
plt.show()

for i in range(14):
    print(names_20[i])

