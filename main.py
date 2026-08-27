#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目启动脚本

此脚本用于快速启动项目和访问各种功能。
"""

import sys
from pathlib import Path


def print_menu():
    """
    打印主菜单。
    """
    print("\n" + "="*70)
    print("🪨 岩石力学中的机器学习应用 (RockMechanics-ML)")
    print("="*70)
    print("\n请选择要执行的操作：")
    print("\n  1. 初始化项目")
    print("  2. 查看项目信息")
    print("  3. 运行岩石强度预测示例")
    print("  4. 查看帮助文档")
    print("  5. 退出")
    print("\n" + "-"*70)


def init_project():
    """
    初始化项目。
    """
    print("\n正在初始化项目...")
    try:
        from init_project import main
        main()
    except Exception as e:
        print(f"初始化失败: {str(e)}")


def show_project_info():
    """
    显示项目信息。
    """
    print("\n" + "="*70)
    print("项目信息")
    print("="*70)
    try:
        from PROJECT_SUMMARY import (
            PROJECT_NAME, PROJECT_VERSION, AUTHOR, DESCRIPTION,
            COMPLETED_MODULES, PROJECT_STATISTICS
        )
        print(f"\n项目名称: {PROJECT_NAME}")
        print(f"版本: {PROJECT_VERSION}")
        print(f"作者: {AUTHOR}")
        print(f"\n描述: {DESCRIPTION}")
        
        print("\n已完成的模块:")
        for category, modules in COMPLETED_MODULES.items():
            print(f"  {category}")
            for module, desc in modules.items():
                print(f"    ✓ {module}")
        
        print("\n项目统计:")
        for key, value in PROJECT_STATISTICS.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"加载项目信息失败: {str(e)}")
    
    print("\n" + "="*70)


def run_example():
    """
    运行示例程序。
    """
    print("\n正在运行岩石强度预测示例...")
    print("(这需要一些时间，请耐心等待)\n")
    try:
        from examples.example_strength_prediction import main
        main()
    except Exception as e:
        print(f"运行示例失败: {str(e)}")
        print("\n提示: 请确保已运行 'python init_project.py' 来初始化项目")


def show_help():
    """
    显示帮助文档。
    """
    print("\n" + "="*70)
    print("帮助文档")
    print("="*70)
    
    help_text = """
📚 文档位置:
  - 安装指南: docs/installation.md
  - 数据格式: docs/data_format.md
  - 使用指南: docs/usage_guide.md
  - 参考资源: docs/references.md

🚀 快速开始:
  1. pip install -r requirements.txt
  2. python init_project.py
  3. python examples/example_strength_prediction.py

📊 支持的模型:
  - Linear Regression (线性回归)
  - Ridge Regression (岭回归)
  - Lasso Regression (套索回归)
  - Support Vector Regression (SVR)
  - Random Forest (随机森林) ⭐ 推荐
  - Gradient Boosting (梯度提升)
  - MLP Neural Network (神经网络)

💡 常用函数:
  - StrengthPredictor: 强度预测器主类
  - StrengthDataLoader: 数据加载器
  - FeatureEngineer: 特征工程工具
  - plot_features(): 特征可视化
  - evaluate_regression(): 模型评估

❓ 常见问题:
  Q: 如何处理缺失值?
  A: 使用 loader.remove_missing_values(method='mean')
  
  Q: 如何选择最好的模型?
  A: 使用交叉验证对比多个模型
  
  Q: 如何提高预测精度?
  A: 进行特征工程、数据清洗、调整超参数

📧 获取支持:
  - GitHub Issues: 报告问题
  - GitHub Discussions: 讨论和建议
  
📖 更多信息: 访问 https://github.com/DongH-L/RockMechanics-ML
    """
    print(help_text)
    print("="*70)


def main():
    """
    主函数。
    """
    while True:
        print_menu()
        choice = input("请输入您的选择 (1-5): ").strip()
        
        if choice == '1':
            init_project()
        elif choice == '2':
            show_project_info()
        elif choice == '3':
            run_example()
        elif choice == '4':
            show_help()
        elif choice == '5':
            print("\n感谢使用 RockMechanics-ML！再见！👋\n")
            sys.exit(0)
        else:
            print("\n❌ 无效的选择，请输入 1-5")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  程序被中止\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序出错: {str(e)}\n")
        sys.exit(1)
