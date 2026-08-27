"""
岩石强度数据加载和预处理模块

提供以下功能：
- CSV 数据加载
- 特征和目标分离
- 数据统计分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Union


class StrengthDataLoader:
    """岩石强度数据加载和预处理工具类。"""
    
    def __init__(self):
        """
        初始化数据加载器。
        
        示例:
        -----
        >>> loader = StrengthDataLoader()
        """
        self.data = None          # 原始数据
        self.features = None      # 特征矩阵
        self.target = None        # 目标向量
        print("✓ 数据加载器已初始化")
    
    def load_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        从 CSV 文件加载数据。
        
        参数:
        -----
        filepath : str 或 Path
            CSV 文件路径
        
        返回:
        -----
        pd.DataFrame
            加载的数据框
        
        示例:
        -----
        >>> loader = StrengthDataLoader()
        >>> df = loader.load_csv('data/rock_strength.csv')
        >>> print(df.shape)  # (样本数, 特征数)
        """
        self.data = pd.read_csv(filepath)
        print(f"✓ 数据已加载: {self.data.shape[0]} 个样本, {self.data.shape[1]} 个特征")
        return self.data
    
    def split_features_target(self, feature_columns: list, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        将数据分离为特征和目标变量。
        
        参数:
        -----
        feature_columns : list
            特征列的列名列表
        target_column : str
            目标列的列名
        
        返回:
        -----
        tuple
            (特征矩阵 shape=(n_samples, n_features), 
             目标向量 shape=(n_samples,))
        
        示例:
        -----
        >>> feature_cols = ['石英含量', '长石含量', '云母含量', '孔隙率', '含水量']
        >>> target_col = '抗压强度'
        >>> X, y = loader.split_features_target(feature_cols, target_col)
        >>> print(f"特征形状: {X.shape}")
        >>> print(f"目标形状: {y.shape}")
        """
        if self.data is None:
            raise ValueError("未加载数据。请先调用 load_csv() 方法。")
        
        # 提取特征和目标
        self.features = self.data[feature_columns].values
        self.target = self.data[target_column].values
        
        print(f"✓ 特征分离完成")
        print(f"  - 特征形状: {self.features.shape}")
        print(f"  - 目标形状: {self.target.shape}")
        
        return self.features, self.target
    
    def get_statistics(self) -> dict:
        """
        获取数据统计信息。
        
        返回:
        -----
        dict
            包含以下统计信息：
            - shape: 数据框形状
            - columns: 列名
            - dtypes: 数据类型
            - missing_values: 缺失值个数
            - describe: 数据统计摘要（均值、std、min、max 等）
        
        示例:
        -----
        >>> stats = loader.get_statistics()
        >>> print(f"缺失值:\n{stats['missing_values']}")
        >>> print(f"统计摘要:\n{stats['describe']}")
        """
        if self.data is None:
            raise ValueError("未加载数据。")
        
        stats = {
            'shape': self.data.shape,  # (行数, 列数)
            'columns': self.data.columns.tolist(),  # 列名列表
            'dtypes': self.data.dtypes.to_dict(),  # 数据类型字典
            'missing_values': self.data.isnull().sum().to_dict(),  # 缺失值计数
            'describe': self.data.describe().to_dict()  # 统计摘要
        }
        
        return stats
    
    def print_statistics(self) -> None:
        """
        打印格式化的数据统计信息。
        
        示例:
        -----
        >>> loader.print_statistics()
        """
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("数据统计信息 (DATA STATISTICS)")
        print("="*60)
        
        print(f"\n数据形状: {stats['shape'][0]} 个样本 × {stats['shape'][1]} 个特征")
        
        print(f"\n数据类型:")
        for col, dtype in stats['dtypes'].items():
            print(f"  {col}: {dtype}")
        
        print(f"\n缺失值 (Missing Values):")
        for col, missing_count in stats['missing_values'].items():
            if missing_count > 0:
                print(f"  {col}: {missing_count}")
        
        print(f"\n统计摘要:")
        for col, stat_dict in stats['describe'].items():
            print(f"  {col}:")
            for stat_name, value in stat_dict.items():
                print(f"    {stat_name}: {value:.4f}")
        
        print("\n" + "="*60)
    
    def remove_missing_values(self, method: str = 'drop') -> pd.DataFrame:
        """
        处理缺失值。
        
        参数:
        -----
        method : str
            处理方法：
            - 'drop': 删除包含缺失值的行（默认）
            - 'mean': 用均值填充
            - 'median': 用中位数填充
        
        返回:
        -----
        pd.DataFrame
            处理后的数据框
        
        示例:
        -----
        >>> df_clean = loader.remove_missing_values(method='mean')
        """
        if self.data is None:
            raise ValueError("未加载数据。")
        
        if method == 'drop':
            self.data = self.data.dropna()
            print(f"✓ 已删除包含缺失值的行，剩余 {len(self.data)} 个样本")
        elif method == 'mean':
            self.data = self.data.fillna(self.data.mean())
            print(f"✓ 已用均值填充缺失值")
        elif method == 'median':
            self.data = self.data.fillna(self.data.median())
            print(f"✓ 已用中位数填充缺失值")
        else:
            raise ValueError(f"未知的处理方法: {method}")
        
        return self.data
