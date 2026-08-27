"""
岩石强度预测示例程序

本示例演示如何使用 RockMechanics-ML 库进行岩石强度预测，包括：
1. 数据加载和探索
2. 特征工程
3. 模型训练
4. 模型评估
5. 结果可视化
"""

import numpy as np
import pandas as pd
from pathlib import Path

# 导入项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strength_prediction import (
    StrengthPredictor,
    StrengthDataLoader,
    FeatureEngineer
)
from utils import (
    split_data,
    plot_features,
    plot_predictions,
    evaluate_regression,
    print_evaluation_report
)


def create_synthetic_data(n_samples: int = 200) -> pd.DataFrame:
    """
    创建合成岩石强度数据用于演示。
    
    参数:
    -----
    n_samples : int
        样本数量
    
    返回:
    -----
    pd.DataFrame
        合成数据框
    """
    print(f"\n生成 {n_samples} 个样本的合成数据...")
    
    np.random.seed(42)
    
    # 生成随机特征
    data = {
        '石英含量': np.random.uniform(40, 60, n_samples),  # %
        '长石含量': np.random.uniform(25, 40, n_samples),  # %
        '云母含量': np.random.uniform(5, 15, n_samples),   # %
        '颗粒大小': np.random.uniform(0.3, 1.0, n_samples),  # mm
        '孔隙率': np.random.uniform(5, 15, n_samples),     # %
        '含水量': np.random.uniform(1, 5, n_samples),      # %
        '风化程度': np.random.randint(1, 6, n_samples),     # 1-5
        '密度': np.random.uniform(2.5, 2.8, n_samples),    # g/cm³
    }
    
    df = pd.DataFrame(data)
    
    # 生成目标变量（模拟真实关系）
    # 强度 = 基础值 + 矿物含量影响 + 孔隙率影响 + 噪声
    strength = (
        100  # 基础抗压强度
        + 0.5 * df['石英含量']
        + 0.3 * df['长石含量']
        - 2 * df['孔隙率']
        - 3 * df['含水量']
        - 10 * df['风化程度']
        + 20 * df['密度']
        + np.random.normal(0, 5, n_samples)  # 噪声
    )
    
    df['抗压强度'] = np.clip(strength, 20, 300)  # 限制在合理范围
    
    print(f"✓ 已生成合成数据")
    return df


def main():
    """
    主函数：完整的岩石强度预测工作流程。
    """
    print("\n" + "="*70)
    print("岩石强度预测示例程序 (ROCK STRENGTH PREDICTION EXAMPLE)")
    print("="*70)
    
    # ========== 第1步：数据加载和探索 ==========
    print("\n[步骤 1] 数据加载和探索")
    print("-" * 70)
    
    # 创建或加载数据
    df = create_synthetic_data(n_samples=200)
    
    # 使用数据加载器
    loader = StrengthDataLoader()
    loader.data = df
    loader.print_statistics()
    
    # ========== 第2步：特征工程 ==========
    print("\n[步骤 2] 特征工程")
    print("-" * 70)
    
    # 分离特征和目标
    feature_cols = ['石英含量', '长石含量', '云母含量', '颗粒大小', 
                    '孔隙率', '含水量', '风化程度', '密度']
    target_col = '抗压强度'
    
    X, y = loader.split_features_target(feature_cols, target_col)
    
    print(f"\n原始特征数: {X.shape[1]}")
    print(f"样本数: {X.shape[0]}")
    
    # 可视化特征分布
    print("\n绘制原始特征分布...")
    plot_features(X, feature_names=feature_cols, figsize=(15, 8))
    
    # 创建交互特征（可选）
    # print("\n创建交互特征...")
    # X_enhanced, feature_names = FeatureEngineer.create_interaction_features(
    #     X, feature_cols
    # )
    # print(f"增强后特征数: {X_enhanced.shape[1]}")
    # X = X_enhanced
    # feature_cols = feature_names
    
    # ========== 第3步：数据分割 ==========
    print("\n[步骤 3] 数据分割")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    print(f"✓ 训练集: {X_train.shape[0]} 样本")
    print(f"✓ 测试集: {X_test.shape[0]} 样本")
    
    # ========== 第4步：模型训练 ==========
    print("\n[步骤 4] 模型训练")
    print("-" * 70)
    
    # 创建多个预测器，比较性能
    models_to_test = [
        ('random_forest', {'n_estimators': 100, 'max_depth': 10}),
        ('gradient_boosting', {'n_estimators': 100, 'learning_rate': 0.1}),
        ('svr', {'kernel': 'rbf', 'C': 100}),
    ]
    
    predictors = {}
    train_results = {}
    
    for model_name, model_kwargs in models_to_test:
        print(f"\n训练 {model_name} 模型...")
        predictor = StrengthPredictor(model_name=model_name, **model_kwargs)
        result = predictor.train(X_train, y_train, scale=True)
        predictors[model_name] = predictor
        train_results[model_name] = result
    
    # ========== 第5步：模型评估 ==========
    print("\n[步骤 5] 模型评估")
    print("-" * 70)
    
    test_results = {}
    best_model_name = None
    best_r2 = -np.inf
    
    for model_name, predictor in predictors.items():
        print(f"\n评估 {model_name} 模型...")
        result = predictor.evaluate(X_test, y_test)
        test_results[model_name] = result
        
        # 找到最佳模型
        if result['R2'] > best_r2:
            best_r2 = result['R2']
            best_model_name = model_name
    
    print(f"\n最佳模型: {best_model_name} (R² = {best_r2:.4f})")
    
    # ========== 第6步：结果可视化 ==========
    print("\n[步骤 6] 结果可视化")
    print("-" * 70)
    
    best_predictor = predictors[best_model_name]
    y_pred = best_predictor.predict(X_test)
    
    print(f"\n绘制预测结果...")
    plot_predictions(y_test, y_pred, title=f'{best_model_name} - 预测值 vs 真实值')
    
    # ========== 第7步：模型保存 ==========
    print("\n[步骤 7] 模型保存")
    print("-" * 70)
    
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f'{best_model_name}_strength_predictor.pkl'
    best_predictor.save(str(model_path))
    
    # ========== 第8步：性能对比 ==========
    print("\n[步骤 8] 模型性能对比")
    print("-" * 70)
    
    print("\n" + "="*70)
    print("训练集性能")
    print("="*70)
    for model_name, metrics in train_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  R² Score: {metrics['R2']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f} MPa")
        print(f"  MAE: {metrics['MAE']:.4f} MPa")
    
    print("\n" + "="*70)
    print("测试集性能")
    print("="*70)
    for model_name, metrics in test_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  R² Score: {metrics['R2']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f} MPa")
        print(f"  MAE: {metrics['MAE']:.4f} MPa")
    
    # ========== 第9步：单个样本预测 ==========
    print("\n[步骤 9] 单个样本预测")
    print("-" * 70)
    
    # 创建一个测试样本
    test_sample = np.array([[
        50,    # 石英含量 (%)
        32,    # 长石含量 (%)
        10,    # 云母含量 (%)
        0.6,   # 颗粒大小 (mm)
        8,     # 孔隙率 (%)
        2,     # 含水量 (%)
        2,     # 风化程度 (1-5)
        2.7    # 密度 (g/cm³)
    ]])
    
    predicted_strength = best_predictor.predict(test_sample)[0]
    
    print(f"\n样本特征:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: {test_sample[0][i]:.2f}")
    
    print(f"\n预测结果:")
    print(f"  预测的抗压强度: {predicted_strength:.2f} MPa")
    
    print("\n" + "="*70)
    print("示例程序完成！")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
