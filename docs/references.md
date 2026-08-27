"""
项目参考文献和资源
"""

# =====================================
# 1. Python 机器学习库
# =====================================

# scikit-learn: https://scikit-learn.org/
# NumPy: https://numpy.org/
# Pandas: https://pandas.pydata.org/
# Matplotlib: https://matplotlib.org/
# TensorFlow: https://www.tensorflow.org/


# =====================================
# 2. 岩石力学相关论文
# =====================================

"""
推荐阅读的学术论文和资源:

1. 岩石强度预测
   - "Machine Learning for Rock Strength Prediction" (2020)
   - "Predicting Uniaxial Compressive Strength Using Machine Learning" (2021)
   
2. 地质灾害预测
   - "Landslide Susceptibility Prediction Using ML Methods" (2021)
   - "Machine Learning for Slope Stability Analysis" (2022)
   
3. 岩爆预测
   - "Deep Learning for Rockburst Prediction" (2021)
   - "Rockburst Risk Assessment Using XGBoost" (2020)
   
4. 岩心图像分析
   - "CNN for Core Image Classification" (2020)
   - "Deep Learning in Digital Rock Physics" (2021)
"""


# =====================================
# 3. 有用的网站和资源
# =====================================

"""
Official Documentation:
- Scikit-learn: https://scikit-learn.org/stable/documentation.html
- Pandas: https://pandas.pydata.org/docs/
- NumPy: https://numpy.org/doc/

Tutorials:
- Kaggle: https://www.kaggle.com/ (数据集和教程)
- Medium: https://medium.com/ (ML 技术文章)
- Towards Data Science: https://towardsdatascience.com/

GitHub 相关项目:
- scikit-learn: https://github.com/scikit-learn/scikit-learn
- XGBoost: https://github.com/dmlc/xgboost

Geology 资源:
- USGS: https://www.usgs.gov/ (美国地质调查局)
- 地球物理学会: 各国地球物理学会官网
"""


# =====================================
# 4. 推荐的学习顺序
# =====================================

"""
初学者学习路径:

第一阶段 (1-2周):
1. 学习 Python 基础 (变量、函数、类)
2. 学习 NumPy (数组操作)
3. 学习 Pandas (数据框操作)
目标: 能够加载和处理数据

第二阶段 (2-3周):
1. 学习 Scikit-learn 基础
2. 学习机器学习基本概念
3. 实践回归和分类问题
目标: 能够训练基础模型

第三阶段 (3-4周):
1. 学习特征工程
2. 学习模型评估和交叉验证
3. 学习超参数调优
目标: 能够优化模型性能

第四阶段 (4周+):
1. 学习深度学习 (TensorFlow)
2. 学习时间序列预测
3. 学习部署和生产化
目标: 构建实用系统
"""


# =====================================
# 5. 数据集来源
# =====================================

"""
公开可用的岩石力学数据集:

1. UCI Machine Learning Repository
   - 地质和地球物理数据集
   - 网址: https://archive.ics.uci.edu/

2. Kaggle Datasets
   - 各类地球科学数据
   - 网址: https://www.kaggle.com/datasets

3. 地质部门官方数据
   - 中国地质调查局数据
   - USGS 数据

4. 学术论文附带数据
   - 从论文的补充材料获取

创建自己的数据集:
- 室内实验（UCS、抗拉强度测试）
- 现场测量（钻孔、地震测量）
- 文献汇总（已发表数据）
"""


# =====================================
# 6. 常见超参数参考值
# =====================================

"""
Random Forest 超参数:
n_estimators: 100-500 (树的数量)
max_depth: 5-20 (树的最大深度)
min_samples_split: 2-10
min_samples_leaf: 1-5
random_state: 42 (重现性)

Gradient Boosting 超参数:
n_estimators: 100-500
learning_rate: 0.01-0.1
max_depth: 3-10
min_samples_split: 2-10
subsample: 0.7-1.0

SVR 超参数:
kernel: 'rbf', 'poly', 'linear'
C: 0.1-1000 (越大正则化越弱)
epsilon: 0.01-0.1 (容错范围)
gamma: 'scale', 'auto', 或 0.001-1

MLP 超参数:
hidden_layer_sizes: (100,), (100, 50), (200, 100, 50)
activation: 'relu', 'tanh'
learning_rate: 0.0001-0.01
max_iter: 500-5000
"""

print("参考文献和资源已加载。")
print("请访问上述网站获取更多信息。")
