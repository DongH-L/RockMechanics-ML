"""
岩石强度预测的回归模型集合

包含以下模型：
- 线性回归 (Linear Regression)
- 岭回归 (Ridge Regression)
- Lasso 回归 (Lasso Regression)
- 支持向量回归 (SVR)
- 随机森林回归 (Random Forest)
- 梯度提升回归 (Gradient Boosting)
- 多层感知机 (MLP Neural Network)
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import joblib
from pathlib import Path


class RegressionModels:
    """回归模型工厂类，提供各种回归模型的创建方法。"""
    
    @staticmethod
    def linear_regression(**kwargs):
        """
        创建线性回归模型。
        
        适用场景：特征与目标呈线性关系
        优点：简单、可解释性强
        缺点：对非线性数据拟合效果差
        
        返回:
        -----
        LinearRegression
            线性回归模型实例
        """
        return LinearRegression(**kwargs)
    
    @staticmethod
    def ridge_regression(alpha=1.0, **kwargs):
        """
        创建岭回归（Ridge）模型。
        
        特点：添加 L2 正则化，防止过拟合
        参数 alpha：正则化强度，越大正则化越强
        
        参数:
        -----
        alpha : float
            正则化参数（默认 1.0）
        
        返回:
        -----
        Ridge
            岭回归模型实例
        """
        return Ridge(alpha=alpha, **kwargs)
    
    @staticmethod
    def lasso_regression(alpha=0.1, **kwargs):
        """
        创建 Lasso 回归模型。
        
        特点：添加 L1 正则化，可进行特征选择
        参数 alpha：正则化强度，越大特征会被更多地淘汰
        
        参数:
        -----
        alpha : float
            正则化参数（默认 0.1）
        
        返回:
        -----
        Lasso
            Lasso 回归模型实例
        """
        return Lasso(alpha=alpha, **kwargs)
    
    @staticmethod
    def support_vector_regression(kernel='rbf', C=100, epsilon=0.1, **kwargs):
        """
        创建支持向量回归（SVR）模型。
        
        特点：适合处理非线性问题，对小规模数据效果好
        内核选项：'linear', 'rbf' (径向基函数), 'poly' (多项式)
        
        参数:
        -----
        kernel : str
            内核类型（默认 'rbf'）
        C : float
            正则化参数（默认 100）
        epsilon : float
            容错率（默认 0.1）
        
        返回:
        -----
        SVR
            支持向量回归模型实例
        """
        return SVR(kernel=kernel, C=C, epsilon=epsilon, **kwargs)
    
    @staticmethod
    def random_forest_regressor(n_estimators=100, max_depth=10, **kwargs):
        """
        创建随机森林回归模型。
        
        特点：集成学习方法，适合处理复杂关系，可处理非线性
        n_estimators：树的数量，越多性能越好但耗时越长
        max_depth：树的最大深度，防止过拟合
        
        参数:
        -----
        n_estimators : int
            决策树数量（默认 100）
        max_depth : int
            树的最大深度（默认 10）
        
        返回:
        -----
        RandomForestRegressor
            随机森林回归模型实例
        """
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                    random_state=42, **kwargs)
    
    @staticmethod
    def gradient_boosting_regressor(n_estimators=100, learning_rate=0.1, **kwargs):
        """
        创建梯度提升回归（GBDT）模型。
        
        特点：序列化树构建，每棵树修正前面的错误
        性能通常比随机森林更好，但训练时间较长
        learning_rate：学习率，控制更新速度
        
        参数:
        -----
        n_estimators : int
            树的数量（默认 100）
        learning_rate : float
            学习率（默认 0.1）
        
        返回:
        -----
        GradientBoostingRegressor
            梯度提升回归模型实例
        """
        return GradientBoostingRegressor(n_estimators=n_estimators, 
                                        learning_rate=learning_rate,
                                        random_state=42, **kwargs)
    
    @staticmethod
    def mlp_regressor(hidden_layer_sizes=(100, 50), max_iter=1000, **kwargs):
        """
        创建多层感知机（神经网络）回归模型。
        
        特点：深度学习方法，适合复杂非线性关系
        hidden_layer_sizes：隐层节点数，如 (100, 50) 表示两层隐层
        max_iter：最大迭代次数
        
        参数:
        -----
        hidden_layer_sizes : tuple
            隐层结构（默认 (100, 50)）
        max_iter : int
            最大迭代次数（默认 1000）
        
        返回:
        -----
        MLPRegressor
            多层感知机回归模型实例
        """
        return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                           max_iter=max_iter, random_state=42, **kwargs)


class StrengthPredictor:
    """岩石强度预测的主要类，包装了完整的训练和预测流程。"""
    
    def __init__(self, model_name: str = 'random_forest', **model_kwargs):
        """
        初始化强度预测器。
        
        参数:
        -----
        model_name : str
            模型类型选择：
            - 'linear': 线性回归
            - 'ridge': 岭回归
            - 'lasso': Lasso 回归
            - 'svr': 支持向量回归
            - 'random_forest': 随机森林（推荐）
            - 'gradient_boosting': 梯度提升
            - 'mlp': 神经网络
        model_kwargs : dict
            传递给模型的其他参数
        
        示例:
        -----
        >>> predictor = StrengthPredictor(model_name='random_forest', n_estimators=200)
        """
        self.model_name = model_name
        self.model = self._build_model(model_name, **model_kwargs)
        self.scaler = StandardScaler()  # 特征标准化器
        self.is_fitted = False  # 标记模型是否已训练
        print(f"✓ 初始化强度预测器 (模型: {model_name})")
    
    def _build_model(self, model_name: str, **kwargs):
        """构建指定类型的回归模型。"""
        models = {
            'linear': RegressionModels.linear_regression,
            'ridge': RegressionModels.ridge_regression,
            'lasso': RegressionModels.lasso_regression,
            'svr': RegressionModels.support_vector_regression,
            'random_forest': RegressionModels.random_forest_regressor,
            'gradient_boosting': RegressionModels.gradient_boosting_regressor,
            'mlp': RegressionModels.mlp_regressor,
        }
        
        if model_name not in models:
            raise ValueError(f"未知的模型类型: {model_name}")
        
        return models[model_name](**kwargs)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              scale: bool = True) -> Dict[str, Any]:
        """
        训练强度预测模型。
        
        参数:
        -----
        X_train : np.ndarray
            训练特征矩阵 (样本数, 特征数)
        y_train : np.ndarray
            训练目标向量 (样本数,)
        scale : bool
            是否对特征进行标准化（推荐 True）
        
        返回:
        -----
        dict
            训练结果，包含 MSE, RMSE, MAE, R2 等指标
        
        示例:
        -----
        >>> train_results = predictor.train(X_train, y_train)
        >>> print(f"R² Score: {train_results['R2']:.4f}")
        """
        print(f"\n开始训练模型...")
        
        # 特征标准化
        if scale:
            X_train_scaled = self.scaler.fit_transform(X_train)
            print(f"✓ 特征已标准化")
        else:
            X_train_scaled = X_train
        
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        print(f"✓ 模型训练完成")
        
        # 获取训练集预测
        y_pred = self.model.predict(X_train_scaled)
        
        # 计算评估指标
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        mse = mean_squared_error(y_train, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        print(f"\n训练结果:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测岩石强度。
        
        参数:
        -----
        X : np.ndarray
            待预测的特征矩阵 (样本数, 特征数)
        
        返回:
        -----
        np.ndarray
            预测的强度值
        
        示例:
        -----
        >>> strength = predictor.predict(X_test)
        >>> print(f"预测的抗压强度: {strength[0]:.2f} MPa")
        """
        if not self.is_fitted:
            raise ValueError("模型未训练。请先调用 train() 方法。")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        在测试集上评估模型性能。
        
        参数:
        -----
        X_test : np.ndarray
            测试特征矩阵
        y_test : np.ndarray
            测试目标向量
        
        返回:
        -----
        dict
            评估指标字典
        
        示例:
        -----
        >>> test_results = predictor.evaluate(X_test, y_test)
        >>> print(f"测试集 R²: {test_results['R2']:.4f}")
        """
        print(f"\n评估模型...")
        
        y_pred = self.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n测试集结果:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def save(self, filepath: str) -> None:
        """
        保存训练好的模型到文件。
        
        参数:
        -----
        filepath : str
            保存路径，建议以 .pkl 结尾
        
        示例:
        -----
        >>> predictor.save('models/strength_predictor.pkl')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_name': self.model_name
        }, filepath)
        print(f"✓ 模型已保存到: {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        从文件加载训练好的模型。
        
        参数:
        -----
        filepath : str
            模型文件路径
        
        示例:
        -----
        >>> predictor = StrengthPredictor()
        >>> predictor.load('models/strength_predictor.pkl')
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_name = data['model_name']
        self.is_fitted = True
        print(f"✓ 模型已加载: {filepath}")
