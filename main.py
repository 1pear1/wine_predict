import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV

#加载了Wine数据集并将其分为特征矩阵X和目标变量y
wine = datasets.load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)
lr = LinearRegression()
lr.fit(X, y)
feature_importances = np.abs(lr.coef_)

#lr.coef_获取线性回归模型的系数，使用np.abs()取绝对值来计算特征的重要性分数
plt.figure(figsize=(8, 4))
bars = plt.bar(X.columns, feature_importances)
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Feature Importance Scores')
plt.xticks(rotation=45)
for bar in bars:
  yval = bar.get_height()
  plt.text(bar.get_x()+bar.get_width()/2, yval, round(yval, 6), ha='center', va='bottom')
plt.show()

# 获取特征与其重要性分数的元组列表，并按照重要性分数排序
feature_scores = list(zip(X.columns, feature_importances))
feature_scores.sort(key=lambda x: x[1], reverse=True)
# 提取前7个特征
top_features = [feature for feature, score in feature_scores[:9]]
# 从特征矩阵X中选择最重要的10个特征
X_top_features = X[top_features]
# 打印选取的特征
print(top_features)

wine = load_wine()
data = pd.DataFrame(X_top_features, columns=top_features)
target = pd.Series(wine.target)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42,stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用PCA进行降维
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(data)

# 创建带有目标变量的完整数据集
df = pd.DataFrame(X_transformed, columns=['PC1', 'PC2'])
df['target'] = target

# 绘制散点图
plt.figure(figsize=(8, 4))
sns.scatterplot(x='PC1', y='PC2', hue='target', data=df, palette='bright')
plt.title('PCA - Wine Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))#以百分比的形式输出准确率

# 定义参数空间
penalty_options = ['none', 'l2', 'l1', 'elasticnet']
solver_options = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# 定义参数网格
param_grid = {'penalty': penalty_options, 'solver': solver_options}

# 创建交叉验证对象
grid_search = GridSearchCV(model, param_grid, cv=5)

# 训练模型并找到最佳参数配置
grid_search.fit(X, y)  # 这里的X和y是你的训练数据和标签

# 输出最佳参数配置
best_params = grid_search.best_params_
print("最佳参数配置:", best_params)

# 使用输出的最佳参数重新预测，并输出准确率
best_model = LogisticRegression(penalty=best_params['penalty'], solver=best_params['solver'])
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print("使用最佳参数重新预测的准确率: {:.2f}%".format(accuracy_best*100))

# 输出混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix:\n", cm)