"""
项目初始化脚本

该脚本用于首次设置项目，创建必要的目录结构和示例数据。
"""

import os
from pathlib import Path
import json


def create_directories():
    """
    创建项目所需的目录结构。
    """
    print("\n创建目录结构...")
    
    directories = [
        'data/raw',                    # 原始数据
        'data/processed',              # 处理后的数据
        'models',                      # 保存的模型
        'results/predictions',         # 预测结果
        'results/visualizations',      # 可视化结果
        'notebooks',                   # Jupyter笔记本
        'logs',                        # 日志文件
        'strength_prediction/models',  # 强度预测模型
        'hazard_warning/models',       # 灾害预警模型
        'rockburst_prediction/models', # 岩爆预测模型
        'core_image_analysis/models',  # 图像分析模型
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ 已创建: {directory}")
    
    print("\n✓ 目录结构创建完成")


def create_config_file():
    """
    创建项目配置文件。
    """
    print("\n创建配置文件...")
    
    config = {
        "project_name": "RockMechanics-ML",
        "version": "0.1.0",
        "author": "DongH-L",
        "description": "Machine Learning Applications in Rock Mechanics",
        "data_paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed"
        },
        "model_paths": {
            "strength_prediction": "models/strength_models",
            "hazard_warning": "models/hazard_warning_models",
            "rockburst_prediction": "models/rockburst_models",
            "core_image_analysis": "models/image_models"
        },
        "output_paths": {
            "predictions": "results/predictions",
            "visualizations": "results/visualizations",
            "logs": "logs"
        },
        "training_params": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5
        },
        "model_configs": {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "random_state": 42
            },
            "svr": {
                "kernel": "rbf",
                "C": 100,
                "epsilon": 0.1
            }
        }
    }
    
    config_path = Path('config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f"✓ 配置文件已创建: {config_path}")


def create_gitignore():
    """
    创建 .gitignore 文件。
    """
    print("\n创建 .gitignore 文件...")
    
    gitignore_content = """# 数据文件
data/raw/
data/processed/
*.csv
*.xlsx
*.xls

# 模型文件
models/
*.pkl
*.h5
*.joblib

# 结果文件
results/
*.png
*.jpg
*.pdf

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# 日志
logs/
*.log

# 环境变量
.env
.env.local

# 其他
.cache/
.pytest_cache/
.coverage
htmlcov/
"""
    
    gitignore_path = Path('.gitignore')
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    print(f"✓ .gitignore 文件已创建: {gitignore_path}")


def create_readme_local():
    """
    创建本地 README 文件。
    """
    print("\n创建本地 README 文件...")
    
    readme_content = """
# 快速开始指南

## 环境配置

### 1. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\\Scripts\\activate  # Windows
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 验证安装
```bash
python -c "import numpy, pandas, sklearn, cv2, tensorflow; print('✓ All packages installed!')"
```

## 项目结构

```
RockMechanics-ML/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   └── processed/                 # 处理后的数据
├── models/                        # 保存的模型
├── results/                       # 结果输出
│   ├── predictions/               # 预测结果
│   └── visualizations/            # 可视化结果
├── strength_prediction/           # 岩石强度预测模块
├── hazard_warning/                # 地质灾害预警模块
├── rockburst_prediction/          # 岩爆预测模块
├── core_image_analysis/           # 岩心图像分析模块
├── utils/                         # 工具函数
├── examples/                      # 示例代码
├── docs/                          # 文档
└── notebooks/                     # Jupyter笔记本
```

## 快速开始

### 1. 岩石强度预测
```python
from strength_prediction import StrengthPredictor, StrengthDataLoader
from utils import split_data, evaluate_regression

# 加载数据
loader = StrengthDataLoader()
df = loader.load_csv('data/rock_strength.csv')

# 分离特征和目标
X, y = loader.split_features_target(feature_cols, 'target_col')

# 分割数据
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# 创建并训练模型
predictor = StrengthPredictor(model_name='random_forest')
predictor.train(X_train, y_train)

# 评估
metrics = predictor.evaluate(X_test, y_test)
print(f"R² Score: {metrics['R2']:.4f}")

# 保存模型
predictor.save('models/strength_model.pkl')
```

### 2. 运行示例程序
```bash
# 完整的岩石强度预测示例
python examples/example_strength_prediction.py

# 快速开始脚本
python notebooks/quick_start.py
```

## 模型选择指南

| 模型 | 适用场景 | 优点 | 缺点 | 推荐 |
|------|--------|------|------|------|
| Linear | 线性关系 | 简单快速 | 非线性拟合差 | ✗ |
| Ridge | 特征多 | 防止过拟合 | 非线性差 | ✓ |
| Lasso | 特征选择 | 自动选择特征 | 计算复杂 | ✓ |
| SVR | 中等规模 | 非线性好 | 调参复杂 | ✓ |
| **Random Forest** | **通用** | **性能好** | **可解释差** | **⭐** |
| Gradient Boosting | 最高精度 | 精度最高 | 易过拟合 | ✓ |
| MLP | 大数据 | 学习能力强 | 需大数据 | ✓ |

## 常见问题

### Q: 如何处理缺失值？
A: 使用 `loader.remove_missing_values(method='mean')`

### Q: 如何标准化数据？
A: 使用 `standardize()` 或 `normalize()` 函数

### Q: 如何选择最佳模型？
A: 使用交叉验证对比多个模型

### Q: 预测结果精度不高？
A: 检查数据质量、特征工程、或调整超参数

## 下一步建议

1. **数据准备**
   - 收集更多真实岩石力学数据
   - 进行数据清洗和特征工程
   - 验证数据质量

2. **模型优化**
   - 尝试不同的算法组合
   - 进行超参数调优
   - 使用集成学习方法

3. **实际应用**
   - 部署模型到生产环境
   - 建立预测系统
   - 进行实地验证

## 技术支持

如有问题，请提交 GitHub Issue 或联系项目维护者。

## 许可证

MIT License

---

祝您使用愉快！🎉
"""
    
    readme_local_path = Path('LOCAL_README.md')
    with open(readme_local_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ 本地 README 已创建: {readme_local_path}")


def main():
    """
    主初始化函数。
    """
    print("\n" + "="*70)
    print("RockMechanics-ML 项目初始化")
    print("="*70)
    
    try:
        # 创建目录结构
        create_directories()
        
        # 创建配置文件
        create_config_file()
        
        # 创建 .gitignore
        create_gitignore()
        
        # 创建本地 README
        create_readme_local()
        
        print("\n" + "="*70)
        print("✓ 项目初始化完成！")
        print("="*70)
        print("\n下一步：")
        print("1. 将您的数据放入 data/raw/ 目录")
        print("2. 查看 LOCAL_README.md 获取快速开始指南")
        print("3. 运行示例: python examples/example_strength_prediction.py")
        print("\n祝您使用愉快！🎉\n")
        
    except Exception as e:
        print(f"\n✗ 初始化失败: {str(e)}")
        return False
    
    return True


if __name__ == '__main__':
    main()
