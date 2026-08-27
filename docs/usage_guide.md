"""
使用指南 (USAGE GUIDE)

本文档详细说明如何使用 RockMechanics-ML 库的各个模块。
"""

# =====================================
# 1. 安装和基本配置
# =====================================

# 安装依赖
# pip install -r requirements.txt

# 或使用 conda
# conda create -n rockmech python=3.9
# conda activate rockmech
# pip install -r requirements.txt


# =====================================
# 2. 数据加载和探索
# =====================================

from strength_prediction import StrengthDataLoader
import pandas as pd

# 创建加载器
loader = StrengthDataLoader()

# 加载 CSV 数据
df = loader.load_csv('data/rock_samples.csv')

# 查看数据统计信息
loader.print_statistics()

# 处理缺失值
df_clean = loader.remove_missing_values(method='mean')


# =====================================
# 3. 数据预处理
# =====================================

from utils import standardize, normalize, remove_outliers
import numpy as np

# 标准化特征 (z-score normalization)
X_scaled, scaler = standardize(X)

# 在测试集上使用相同的缩放器
X_test_scaled = standardize(X_test, scaler)

# 归一化特征 (0-1 范围)
X_normalized, scaler = normalize(X)

# 移除离群值 (使用 IQR 方法)
X_clean = remove_outliers(X, method='iqr', threshold=1.5)


# =====================================
# 4. 特征工程
# =====================================

from strength_prediction import FeatureEngineer

# 创建交互特征
X_with_interactions, feature_names = FeatureEngineer.create_interaction_features(
    X, feature_names=['石英含量', '长石含量', '孔隙率', '含水量']
)

# 创建多项式特征
X_poly, poly_feature_names = FeatureEngineer.create_polynomial_features(
    X, degree=2, feature_names=['石英含量', '长石含量']
)

# 归一化矿物含量
df_normalized = FeatureEngineer.normalize_mineral_content(
    df, ['石英含量', '长石含量', '云母含量']
)

# 编码分类特征
df_encoded, encoding_info = FeatureEngineer.encode_categorical(
    df, ['岩石类型', '风化程度']
)


# =====================================
# 5. 模型训练和评估
# =====================================

from strength_prediction import StrengthPredictor
from utils import split_data, evaluate_regression, print_evaluation_report

# 分割数据
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# 创建预测器
predictor = StrengthPredictor(
    model_name='random_forest',  # 可选: 'linear', 'ridge', 'lasso', 'svr', 'random_forest', 'gradient_boosting', 'mlp'
    n_estimators=100,
    max_depth=10
)

# 训练模型
train_results = predictor.train(X_train, y_train, scale=True)

# 评估模型
test_results = predictor.evaluate(X_test, y_test)

# 进行预测
y_pred = predictor.predict(X_test)

# 计算详细指标
metrics = evaluate_regression(y_test, y_pred)
print_evaluation_report(metrics)


# =====================================
# 6. 结果可视化
# =====================================

from utils import (
    plot_features,
    plot_predictions,
    plot_confusion_matrix,
    plot_training_history,
    save_plot
)

# 绘制特征分布
plot_features(X, feature_names=['石英含量', '长石含量', '孔隙率', '含水量'])

# 绘制预测值 vs 真实值
plot_predictions(y_test, y_pred, title='岩石强度预测结果')

# 保存图形
save_plot('results/predictions.png', dpi=300)


# =====================================
# 7. 模型保存和加载
# =====================================

# 保存模型
predictor.save('models/strength_model.pkl')

# 加载模型
loaded_predictor = StrengthPredictor()
loaded_predictor.load('models/strength_model.pkl')

# 使用加载的模型进行预测
predictions = loaded_predictor.predict(X_new)


# =====================================
# 8. 交叉验证
# =====================================

from utils import cross_validate

# 5 折交叉验证
cv_results = cross_validate(predictor.model, X, y, cv=5, scoring='r2')

print(f"交叉验证平均分: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
print(f"各折得分: {cv_results['scores']}")


# =====================================
# 9. 常用的模型类型选择
# =====================================

"""
不同的模型适用于不同的场景：

1. LINEAR (线性回归)
   - 适用: 特征与目标呈线性关系的问题
   - 优点: 简单，可解释性强
   - 缺点: 对非线性关系拟合效果差
   - 使用: 用于基线模型或快速实验

2. RIDGE (岭回归)
   - 适用: 特征数多，可能存在多重共线性
   - 优点: 防止过拟合，稳定性好
   - 缺点: 对强非线性关系拟合效果差
   - 使用: 当有大量特征时

3. LASSO (Lasso回归)
   - 适用: 需要进行特征选择的问题
   - 优点: 能自动进行特征选择
   - 缺点: 计算复杂度较高
   - 使用: 需要找出最重要特征时

4. SVR (支持向量回归)
   - 适用: 中等规模数据，非线性关系
   - 优点: 对非线性问题效果好，鲁棒性强
   - 缺点: 训练速度较慢，超参数调优复杂
   - 使用: 数据量 100-10000 样本时

5. RANDOM_FOREST (随机森林) ⭐ 推荐
   - 适用: 大多数回归问题（推荐首选）
   - 优点: 性能好，能处理非线性，鲁棒性强
   - 缺点: 模型复杂度高，可解释性较差
   - 使用: 一般性能最好，适合岩石力学问题

6. GRADIENT_BOOSTING (梯度提升)
   - 适用: 需要最高精度的问题
   - 优点: 性能通常最好，精度高
   - 缺点: 训练时间长，易过拟合
   - 使用: 数据充分且有计算能力时

7. MLP (神经网络)
   - 适用: 大规模数据，复杂非线性问题
   - 优点: 能学习复杂模式
   - 缺点: 需要大量数据，训练时间长
   - 使用: 数据量 > 1000 样本时

推荐顺序：
1. 从 random_forest 开始
2. 如果性能不足，尝试 gradient_boosting
3. 如果需要简单可解释，使用 svr 或 ridge
"""


# =====================================
# 10. 常见问题解决
# =====================================

"""
Q1: 如何处理缺失值？
A: ��用 loader.remove_missing_values(method='mean') 或 'median'

Q2: 特征之间单位不同怎么办？
A: 使用 standardize() 或 normalize() 进行标准化

Q3: 如何选择最好的模型？
A: 使用交叉验证对比多个模型，选择 R² 最高的

Q4: 如何防止过拟合？
A: 1. 使用正则化模型 (Ridge, Lasso)
   2. 减少特征数量
   3. 增加训练样本
   4. 调整模型复杂度参数

Q5: 预测结果精度不高怎么办？
A: 1. 检查数据质量和是否有离群值
   2. 尝试特征工程（交互特征、多项式特征）
   3. 调整模型超参数
   4. 尝试其他模型
   5. 收集更多数据
"""

print("使用指南已加载，请参考上述示例进行操作。")
