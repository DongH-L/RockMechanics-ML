"""
岩石强度预测的特征工程模块

提供特征工程工具包括：
- 交互特征创建
- 多项式特征生成
- 矿物含量归一化
- 分类特征编码
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


class FeatureEngineer:
    """特征工程工具类，用于创建和转换特征。"""
    
    @staticmethod
    def create_interaction_features(X: np.ndarray, feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        创建交互特征（两个特征的乘积）。
        
        示例：如果有特征 [孔隙率, 含水量]，会生成额外特征 [孔隙率 × 含水量]
        
        参数:
        -----
        X : np.ndarray
            特征矩阵
        feature_names : list, 可选
            特征名称列表
        
        返回:
        -----
        tuple
            (增强的特征矩阵, 更新的特征名称列表)
        
        示例:
        -----
        >>> X = np.array([[8.3, 2.1], [9.1, 2.5]])
        >>> names = ['孔隙率', '含水量']
        >>> X_enhanced, new_names = FeatureEngineer.create_interaction_features(X, names)
        """
        n_samples, n_features = X.shape
        interactions = []
        interaction_names = []
        
        # 生成所有特征对的乘积
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction = X[:, i] * X[:, j]
                interactions.append(interaction)
                
                if feature_names:
                    interaction_names.append(f"{feature_names[i]} × {feature_names[j]}")
                else:
                    interaction_names.append(f"特征_{i} × 特征_{j}")
        
        interactions = np.column_stack(interactions)
        X_enhanced = np.hstack([X, interactions])
        
        if feature_names:
            names = feature_names + interaction_names
        else:
            names = None
        
        print(f"✓ 创建了 {len(interaction_names)} 个交互特征")
        return X_enhanced, names
    
    @staticmethod
    def create_polynomial_features(X: np.ndarray, degree: int = 2, 
                                   feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        创建多项式特征。
        
        示例：对于特征 [x1, x2]，degree=2 会生成 [1, x1, x2, x1², x1×x2, x2²]
        
        参数:
        -----
        X : np.ndarray
            特征矩阵
        degree : int
            多项式次数（默认 2）
        feature_names : list, 可选
            特征名称列表
        
        返回:
        -----
        tuple
            (多项式特征矩阵, 特征名称列表)
        
        示例:
        -----
        >>> X = np.array([[8.3, 2.1], [9.1, 2.5]])
        >>> names = ['孔隙率', '含水量']
        >>> X_poly, poly_names = FeatureEngineer.create_polynomial_features(X, degree=2, feature_names=names)
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        if feature_names:
            names = poly.get_feature_names_out(feature_names).tolist()
        else:
            names = None
        
        print(f"✓ 创建了 {X_poly.shape[1] - X.shape[1]} 个多项式特征 (degree={degree})")
        return X_poly, names
    
    @staticmethod
    def normalize_mineral_content(df: pd.DataFrame, mineral_columns: List[str]) -> pd.DataFrame:
        """
        归一化矿物含量，使其总和为 100%。
        
        用途：确保矿物成分百分比的一致性
        
        参数:
        -----
        df : pd.DataFrame
            输入数据框
        mineral_columns : list
            包含矿物百分比的列名列表
        
        返回:
        -----
        pd.DataFrame
            矿物含量已归一化的数据框
        
        示例:
        -----
        >>> df = pd.DataFrame({
        ...     '石英': [45, 48],
        ...     '长石': [32, 28],
        ...     '云母': [8, 9]
        ... })
        >>> df_norm = FeatureEngineer.normalize_mineral_content(
        ...     df, ['石英', '长石', '云母']
        ... )
        """
        df_copy = df.copy()
        total = df_copy[mineral_columns].sum(axis=1)
        
        # 将每个矿物的百分比除以总和，再乘以 100
        for col in mineral_columns:
            df_copy[col] = (df_copy[col] / total) * 100
        
        print(f"✓ 已归一化 {len(mineral_columns)} 个矿物含量")
        return df_copy
    
    @staticmethod
    def encode_categorical(df: pd.DataFrame, categorical_columns: List[str]) -> Tuple[pd.DataFrame, dict]:
        """
        对分类特征进行独热编码（One-Hot Encoding）。
        
        用途：将分类变量转换为数值变量，供机器学习模型使用
        
        参数:
        -----
        df : pd.DataFrame
            输入数据框
        categorical_columns : list
            分类列的名称列表
        
        返回:
        -----
        tuple
            (编码后的数据框, 编码映射字典)
        
        示例:
        -----
        >>> df = pd.DataFrame({
        ...     '岩石类型': ['花岗岩', '玄武岩', '花岗岩'],
        ...     '风化程度': ['新鲜', '微风化', '新鲜']
        ... })
        >>> df_encoded, encoding = FeatureEngineer.encode_categorical(
        ...     df, ['岩石类型', '风化程度']
        ... )
        """
        # 使用 pandas 的 get_dummies 进行独热编码
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        # 保存编码信息
        encoding_info = {}
        for col in categorical_columns:
            encoding_info[col] = df[col].unique().tolist()
        
        print(f"✓ 已编码 {len(categorical_columns)} 个分类特征")
        return df_encoded, encoding_info
    
    @staticmethod
    def select_features_by_importance(model, feature_names: List[str], top_k: int = 10) -> List[str]:
        """
        根据模型的特征重要性选择最重要的特征。
        
        适用于：树型模型（随机森林、梯度提升等）
        
        参数:
        -----
        model : object
            已训练的模型（需要有 feature_importances_ 属性）
        feature_names : list
            特征名称列表
        top_k : int
            选择前 K 个重要特征
        
        返回:
        -----
        list
            最重要的特征名称列表
        
        示例:
        -----
        >>> important_features = FeatureEngineer.select_features_by_importance(
        ...     model, feature_names, top_k=5
        ... )
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("模型不支持特征重要性分析")
        
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 按重要性排序
        indices = np.argsort(importances)[::-1][:top_k]
        
        # 返回最重要的特征名称
        top_features = [feature_names[i] for i in indices]
        
        print(f"✓ 前 {top_k} 个重要特征:")
        for i, (feat, imp) in enumerate(zip(top_features, importances[indices])):
            print(f"  {i+1}. {feat}: {imp:.4f}")
        
        return top_features
