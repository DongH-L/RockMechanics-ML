"""数据预处理工具模块"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Union, Tuple


def standardize(X: np.ndarray, scaler=None) -> Union[np.ndarray, Tuple[np.ndarray, StandardScaler]]:
    """
    使用 z-score 标准化特征。
    
    标准化公式: (x - mean) / std
    
    参数:
    -----
    X : np.ndarray
        特征矩阵
    scaler : StandardScaler, 可选
        预拟合的缩放器（用于测试数据）
    
    返回:
    -----
    np.ndarray 或 tuple
        标准化后的特征或 (特征, 缩放器)
    
    示例:
    -----
    # 训练集：返回标准化特征和缩放器
    >>> X_train_scaled, scaler = standardize(X_train)
    # 测试集：使用同一个缩放器
    >>> X_test_scaled = standardize(X_test, scaler)
    """
    if scaler is None:
        # 创建新的缩放器
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        # 使用已有的缩放器
        return scaler.transform(X)


def normalize(X: np.ndarray, scaler=None) -> Union[np.ndarray, Tuple[np.ndarray, MinMaxScaler]]:
    """
    将特征归一化到 [0, 1] 范围。
    
    归一化公式: (x - min) / (max - min)
    
    参数:
    -----
    X : np.ndarray
        特征矩阵
    scaler : MinMaxScaler, 可选
        预拟合的缩放器（用于测试数据）
    
    返回:
    -----
    np.ndarray 或 tuple
        归一化后的特征或 (特征, 缩放器)
    
    示例:
    -----
    >>> X_train_norm, scaler = normalize(X_train)
    >>> X_test_norm = normalize(X_test, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        return scaler.transform(X)


def robust_scale(X: np.ndarray, scaler=None) -> Union[np.ndarray, Tuple[np.ndarray, RobustScaler]]:
    """
    鲁棒缩放，使用中位数和四分位间距（对离群值不敏感）。
    
    公式: (x - median) / IQR
    
    参数:
    -----
    X : np.ndarray
        特征矩阵
    scaler : RobustScaler, 可选
        预拟合的缩放器（用于测试数据）
    
    返回:
    -----
    np.ndarray 或 tuple
        缩放后的特征或 (特征, 缩放器)
    
    示例:
    -----
    >>> X_robust, scaler = robust_scale(X)
    """
    if scaler is None:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        return scaler.transform(X)


def handle_missing_values(df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
    """
    处理缺失值。
    
    参数:
    -----
    df : pd.DataFrame
        输入数据框
    method : str
        处理方法:
        - 'drop': 删除包含缺失值的行
        - 'mean': 用平均值填充
        - 'median': 用中位数填充
        - 'forward_fill': 向前填充
    
    返回:
    -----
    pd.DataFrame
        处理后的数据框
    
    示例:
    -----
    >>> df_clean = handle_missing_values(df, method='mean')
    """
    if method == 'drop':
        return df.dropna()
    elif method == 'mean':
        return df.fillna(df.mean())
    elif method == 'median':
        return df.fillna(df.median())
    elif method == 'forward_fill':
        return df.fillna(method='ffill')
    else:
        raise ValueError(f"未知的处理方法: {method}")


def remove_outliers(X: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
    """
    移除离群值。
    
    参数:
    -----
    X : np.ndarray
        特征矩阵
    method : str
        检测方法:
        - 'iqr': 四分位间距法
        - 'zscore': z-score 法
    threshold : float
        检测阈值
        - iqr 方法: 通常为 1.5（外离群值）或 3.0（极端离群值）
        - zscore 方法: 通常为 3（标准差倍数）
    
    返回:
    -----
    np.ndarray
        移除离群值后的数据
    
    示例:
    -----
    >>> X_clean = remove_outliers(X, method='iqr', threshold=1.5)
    """
    if method == 'iqr':
        # 四分位间距法 (IQR)
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        # 识别异常值：小于 Q1-1.5*IQR 或大于 Q3+1.5*IQR
        mask = ~((X < (Q1 - threshold * IQR)) | (X > (Q3 + threshold * IQR))).any(axis=1)
        return X[mask]
    elif method == 'zscore':
        # z-score 法
        from scipy import stats
        z_scores = np.abs(stats.zscore(X))
        # 保留所有特征的 z-score 都小于阈值的样本
        mask = (z_scores < threshold).all(axis=1)
        return X[mask]
    else:
        raise ValueError(f"未知的检测方法: {method}")
