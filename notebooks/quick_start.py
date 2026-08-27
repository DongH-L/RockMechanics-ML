"""
快速开始指南

本指南展示如何快速开始使用 RockMechanics-ML 库进行岩石强度预测。
"""

# 导入必要的库
from strength_prediction import StrengthPredictor, StrengthDataLoader
from utils import split_data, evaluate_regression, print_evaluation_report
import numpy as np
import pandas as pd


# ============ 方式一：使用自己的数据 ============

# 第1步：加载数据
loader = StrengthDataLoader()
df = loader.load_csv('data/rock_strength.csv')

# 第2步：分离特征和目标
feature_cols = ['石英含量', '长石含量', '云母含量', '颗粒大小', 
                '孔隙率', '含水量', '风化程度', '密度']
X, y = loader.split_features_target(feature_cols, '抗压强度')

# 第3步：分割数据
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# 第4步：创建并训练模型
predictor = StrengthPredictor(model_name='random_forest', n_estimators=100)
train_metrics = predictor.train(X_train, y_train, scale=True)

# 第5步：评估模型
test_metrics = predictor.evaluate(X_test, y_test)

# 第6步：进行预测
new_sample = np.array([[50, 32, 10, 0.6, 8, 2, 2, 2.7]])
strength_prediction = predictor.predict(new_sample)
print(f"预测强度: {strength_prediction[0]:.2f} MPa")

# 第7步：保存模型
predictor.save('models/my_strength_predictor.pkl')


# ============ 方式二：快速预测（使用已保存的模型） ============

# 加载模型
loaded_predictor = StrengthPredictor()
loaded_predictor.load('models/my_strength_predictor.pkl')

# 进行预测
new_samples = np.array([
    [50, 32, 10, 0.6, 8, 2, 2, 2.7],
    [48, 35, 9, 0.5, 7, 1.5, 1, 2.75],
])
predictions = loaded_predictor.predict(new_samples)

print("\n预测结果:")
for i, pred in enumerate(predictions):
    print(f"样本 {i+1}: {pred:.2f} MPa")


# ============ 方式三：多模型对比 ============

model_names = ['linear', 'ridge', 'svr', 'random_forest', 'gradient_boosting']
results = {}

for model_name in model_names:
    # 创建模型
    predictor = StrengthPredictor(model_name=model_name)
    
    # 训练
    train_metrics = predictor.train(X_train, y_train)
    
    # 评估
    test_metrics = predictor.evaluate(X_test, y_test)
    
    results[model_name] = test_metrics
    print(f"{model_name}: R² = {test_metrics['R2']:.4f}")


# ============ 方式四：特征工程 ============

from strength_prediction import FeatureEngineer

# 创建交互特征
X_enhanced, new_feature_names = FeatureEngineer.create_interaction_features(
    X, feature_names=feature_cols
)

print(f"\n原始特征数: {X.shape[1]}")
print(f"增强后特征数: {X_enhanced.shape[1]}")

# 使用增强特征训练模型
X_train_enhanced, X_test_enhanced, _, _ = split_data(
    X_enhanced, y, test_size=0.2, random_state=42
)

predictor_enhanced = StrengthPredictor(model_name='random_forest')
predictor_enhanced.train(X_train_enhanced, y_train)
metrics_enhanced = predictor_enhanced.evaluate(X_test_enhanced, y_test)

print(f"增强模型 R²: {metrics_enhanced['R2']:.4f}")
