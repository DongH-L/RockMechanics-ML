"""
岩石强度预测模块初始化

本模块提供了岩石强度预测的完整工具链，包括：
- 数据加载和预处理
- 特征工程
- 多种回归模型
- 模型训练和预测
"""

from .models import StrengthPredictor, RegressionModels
from .feature_engineering import FeatureEngineer
from .data_loader import StrengthDataLoader

__all__ = [
    'StrengthPredictor',        # 强度预测器主类
    'RegressionModels',          # 回归模型集合
    'FeatureEngineer',           # 特征工程工具
    'StrengthDataLoader',        # 数据加载器
]
