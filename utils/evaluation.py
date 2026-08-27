"""模型评估工具模块"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import pandas as pd


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    评估回归模型的性能。
    
    参数:
    -----
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    
    返回:
    -----
    dict
        评估指标: MSE, RMSE, MAE, R2, MAPE
    
    指标说明:
    --------
    - MSE (Mean Squared Error): 均方误差，值越小越好
    - RMSE (Root Mean Squared Error): 均方根误差，与原始数据同单位
    - MAE (Mean Absolute Error): 平均绝对误差，更鲁棒
    - R2 Score: 决定系数，范围 [0, 1]，越接近 1 越好
    - MAPE (Mean Absolute Percentage Error): 平均绝对百分比误差
    
    示例:
    -----
    >>> metrics = evaluate_regression(y_true, y_pred)
    >>> print(f"R2 Score: {metrics['R2']:.4f}")
    """
    # 计算各项指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # 均方根误差
    mae = mean_absolute_error(y_true, y_pred)  # 平均绝对误差
    r2 = r2_score(y_true, y_pred)  # 决定系数
    
    # 计算平均绝对百分比误差 (MAPE)
    # 避免除以零
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> dict:
    """
    评估分类模型的性能。
    
    参数:
    -----
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测标签
    y_pred_proba : np.ndarray, 可选
        预测概率（用于计算 ROC-AUC）
    
    返回:
    -----
    dict
        评估指标: Accuracy, Precision, Recall, F1, Confusion Matrix
    
    指标说明:
    --------
    - Accuracy: 准确率，所有样本中预测正确的比例
    - Precision: 精准率，预测正例中实际正例的比例
    - Recall: 召回率，实际正例中预测正确的比例
    - F1: 精准率和召回率的调和平均数
    - Confusion Matrix: 混淆矩阵，显示分类结果的详细情况
    
    示例:
    -----
    >>> metrics = evaluate_classification(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['Accuracy']:.4f}")
    """
    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Confusion_Matrix': cm
    }
    
    # 如果有预测概率且是二分类，添加 ROC-AUC
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            results['ROC-AUC'] = roc_auc
        except:
            pass
    
    return results


def cross_validate(model, X: np.ndarray, y: np.ndarray, cv: int = 5, 
                   scoring: str = None) -> dict:
    """
    执行 k 折交叉验证。
    
    参数:
    -----
    model : object
        具有 fit 和 predict 方法的 sklearn 模型
    X : np.ndarray
        特征矩阵
    y : np.ndarray
        目标向量
    cv : int
        折数 (默认 5)
    scoring : str, 可选
        评分指标
    
    返回:
    -----
    dict
        交叉验证分数
    
    示例:
    -----
    >>> cv_results = cross_validate(model, X, y, cv=5, scoring='r2')
    >>> print(f"平均分数: {cv_results['mean']:.4f}")
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'scores': scores,  # 所有折的分数
        'mean': scores.mean(),  # 平均分数
        'std': scores.std(),  # 标准差
        'fold_scores': {f'fold_{i}': score for i, score in enumerate(scores)}  # 按折显示
    }


def print_evaluation_report(metrics: dict) -> None:
    """
    以格式化的方式打印评估指标。
    
    参数:
    -----
    metrics : dict
        评估指标字典
    
    示例:
    -----
    >>> print_evaluation_report(metrics)
    """
    print("\n" + "="*60)
    print("模型评估报告 (MODEL EVALUATION REPORT)")
    print("="*60)
    
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            # 数组类型（如混淆矩阵）
            print(f"\n{key}:")
            print(value)
        elif isinstance(value, dict):
            # 字典类型
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            # 标量值
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    print("\n" + "="*60)
