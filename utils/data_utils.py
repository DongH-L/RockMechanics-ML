"""数据加载和处理工具模块"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Union
import pickle
import json


def load_data(filepath: Union[str, Path], format: str = 'csv') -> pd.DataFrame:
    """
    从文件加载数据。
    
    参数:
    -----
    filepath : str 或 Path
        数据文件路径
    format : str
        数据格式: 'csv', 'excel', 'json'
    
    返回:
    -----
    pd.DataFrame
        加载的数据
    
    示例:
    -----
    >>> df = load_data('data.csv', format='csv')
    """
    filepath = Path(filepath)
    
    if format.lower() == 'csv':
        return pd.read_csv(filepath)
    elif format.lower() in ['excel', 'xlsx', 'xls']:
        return pd.read_excel(filepath)
    elif format.lower() == 'json':
        return pd.read_json(filepath)
    else:
        raise ValueError(f"不支持的格式: {format}")


def save_data(data: pd.DataFrame, filepath: Union[str, Path], format: str = 'csv') -> None:
    """
    将数据保存到文件。
    
    参数:
    -----
    data : pd.DataFrame
        要保存的数据
    filepath : str 或 Path
        输出文件路径
    format : str
        数据格式: 'csv', 'excel', 'json'
    
    示例:
    -----
    >>> save_data(df, 'output.csv', format='csv')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'csv':
        data.to_csv(filepath, index=False)
    elif format.lower() in ['excel', 'xlsx', 'xls']:
        data.to_excel(filepath, index=False)
    elif format.lower() == 'json':
        data.to_json(filepath)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    print(f"数据已保存到: {filepath}")


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将数据分割为训练集和测试集。
    
    参数:
    -----
    X : np.ndarray
        特征矩阵 (样本数, 特征数)
    y : np.ndarray
        目标向量 (样本数,)
    test_size : float
        测试集比例 (0.0-1.0)
    random_state : int
        随机种子，用于可重复性
    
    返回:
    -----
    tuple
        (X_train, X_test, y_train, y_test)
    
    示例:
    -----
    >>> X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def save_model(model, filepath: Union[str, Path]) -> None:
    """
    将训练好的模型保存到文件。
    
    参数:
    -----
    model : object
        训练好的模型对象
    filepath : str 或 Path
        输出文件路径
    
    示例:
    -----
    >>> save_model(predictor, 'models/strength_model.pkl')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"模型已保存到: {filepath}")


def load_model(filepath: Union[str, Path]):
    """
    从文件加载训练好的模型。
    
    参数:
    -----
    filepath : str 或 Path
        模型文件路径
    
    返回:
    -----
    object
        加载的模型对象
    
    示例:
    -----
    >>> predictor = load_model('models/strength_model.pkl')
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"模型已加载: {filepath}")
    return model
