"""数据可视化工具模块"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
from typing import Optional, Union

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def plot_features(X: np.ndarray, feature_names: list = None, figsize: tuple = (12, 6)) -> None:
    """
    绘制特征分布直方图。
    
    参数:
    -----
    X : np.ndarray
        特征矩阵
    feature_names : list, 可选
        特征名称
    figsize : tuple
        图形大小 (宽, 高)
    
    示例:
    -----
    >>> plot_features(X, feature_names=['石英含量', '长石含量', '孔隙率'])
    """
    n_features = X.shape[1]
    n_cols = 3  # 每行 3 个子图
    n_rows = (n_features + n_cols - 1) // n_cols  # 计算所需行数
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]  # 展平为 1D 数组
    
    # 绘制每个特征的分布
    for i in range(n_features):
        axes[i].hist(X[:, i], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        title = feature_names[i] if feature_names else f'特征 {i}'
        axes[i].set_title(title, fontsize=10)
        axes[i].set_xlabel('数值')
        axes[i].set_ylabel('频数')
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    title: str = '预测值 vs 真实值',
                    figsize: tuple = (8, 6)) -> None:
    """
    绘制预测值与真实值的散点图。
    
    参数:
    -----
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    title : str
        图标题
    figsize : tuple
        图形大小
    
    示例:
    -----
    >>> plot_predictions(y_test, y_pred, title='抗压强度预测')
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=50)
    
    # 绘制完美预测线（y=x）
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
    
    plt.xlabel('真实值', fontsize=11)
    plt.ylabel('预测值', fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: list = None,
                         figsize: tuple = (8, 6)) -> None:
    """
    绘制混淆矩阵热力图。
    
    参数:
    -----
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测标签
    class_names : list, 可选
        类别名称
    figsize : tuple
        图形大小
    
    示例:
    -----
    >>> classes = ['安全', '注意', '警告', '危险']
    >>> plot_confusion_matrix(y_test, y_pred, class_names=classes)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '数量'})
    plt.ylabel('真实标签', fontsize=11)
    plt.xlabel('预测标签', fontsize=11)
    plt.title('混淆矩阵', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict, figsize: tuple = (12, 4)) -> None:
    """
    绘制训练历史（损失和准确率）。
    
    参数:
    -----
    history : dict
        训练历史字典，包含 'loss', 'val_loss', 'accuracy', 'val_accuracy'
    figsize : tuple
        图形大小
    
    示例:
    -----
    >>> plot_training_history({
    ...     'loss': [...],
    ...     'val_loss': [...],
    ...     'accuracy': [...],
    ...     'val_accuracy': [...]
    ... })
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 绘制损失曲线
    if 'loss' in history:
        axes[0].plot(history['loss'], label='训练损失', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_xlabel('迭代次数 (Epoch)', fontsize=11)
    axes[0].set_ylabel('损失值', fontsize=11)
    axes[0].set_title('模型损失', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='训练准确率', linewidth=2)
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='验证准确率', linewidth=2)
    axes[1].set_xlabel('迭代次数 (Epoch)', fontsize=11)
    axes[1].set_ylabel('准确率', fontsize=11)
    axes[1].set_title('模型准确率', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def save_plot(filepath: Union[str, Path], dpi: int = 300) -> None:
    """
    保存当前图形到文件。
    
    参数:
    -----
    filepath : str 或 Path
        输出文件路径
    dpi : int
        分辨率 (dots per inch)
    
    示例:
    -----
    >>> plot_features(X)
    >>> save_plot('results/feature_distribution.png', dpi=300)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"图形已保存到: {filepath}")
