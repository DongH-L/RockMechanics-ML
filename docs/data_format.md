# 数据格式说明

本文档详细说明了各个模块所需的数据格式和结构。

---

## 1. 岩石强度预测数据 (Strength Prediction)

### 输入特征 (Features)

| 特征名 | 数据类型 | 单位 | 说明 |
|---|---|---|---|
| quartz_content | float | % | 石英含量 |
| feldspar_content | float | % | 长石含量 |
| mica_content | float | % | 云母含量 |
| grain_size | float | mm | 平均颗粒大小 |
| porosity | float | % | 孔隙率 |
| water_content | float | % | 含水量 |
| weathering_degree | int | 1-5 | 风化程度 (1=新鲜, 5=完全风化) |
| uniaxial_compression_strength | float | MPa | 单轴抗压强度 |
| density | float | g/cm³ | 岩石密度 |

### 目标变量 (Target)

```
强度指标：
- uniaxial_compressive_strength (UCS): 单轴抗压强度 (MPa)
- tensile_strength: 抗拉强度 (MPa)
- splitting_tensile_strength: 劈裂抗拉强度 (MPa)
```

### 数据格式示例

**CSV 格式:**
```csv
sample_id,quartz_content,feldspar_content,mica_content,grain_size,porosity,water_content,weathering_degree,density,uniaxial_compressive_strength
S001,45.5,32.1,8.2,0.5,8.3,2.1,2,2.65,120.5
S002,48.2,28.5,9.1,0.6,9.1,2.5,2,2.63,118.2
S003,42.1,35.2,7.5,0.4,7.8,1.9,1,2.68,132.1
...
```

**Python 读取:**
```python
import pandas as pd
from strength_prediction import StrengthPredictor

# 读取数据
df = pd.read_csv('rock_strength_data.csv')

# 分离特征和目标
X = df[['quartz_content', 'feldspar_content', 'mica_content', 
        'grain_size', 'porosity', 'water_content', 'weathering_degree', 'density']]
y = df['uniaxial_compressive_strength']

# 训练模型
predictor = StrengthPredictor()
predictor.train(X, y)
```

---

## 2. 地质灾害预警数据 (Hazard Warning)

### 输入特征 (Slope Parameters)

| 特征名 | 数据类型 | 单位 | 说明 |
|---|---|---|---|
| slope_angle | float | ° | 边坡坡角 |
| slope_height | float | m | 边坡高度 |
| soil_cohesion | float | kPa | 土壤粘聚力 |
| internal_friction_angle | float | ° | 内摩擦角 |
| water_table_depth | float | m | 地下水位深度 |
| rainfall_intensity | float | mm/h | 降雨强度 |
| vegetation_coverage | float | % | 植被覆盖度 |
| slope_stability_factor | float | - | 安全系数（已知时） |

### 目标变量 (Risk Level)

```
风险等级分类（多分类）:
- 0: 安全 (Stable)
- 1: 注意 (Alert)
- 2: 警告 (Warning)
- 3: 危险 (Danger)
```

### 数据格式示例

**CSV 格式:**
```csv
site_id,date,slope_angle,slope_height,soil_cohesion,internal_friction_angle,water_table_depth,rainfall_intensity,vegetation_coverage,risk_level
S001,2024-01-15,35.5,50,25.3,32.1,5.2,2.1,60,0
S002,2024-01-15,42.1,75,18.5,28.3,2.1,5.5,35,1
S003,2024-01-15,48.2,120,15.2,25.1,1.5,8.3,20,2
...
```

**实时预警示例:**
```python
from hazard_warning import HazardAlertSystem
import numpy as np

# 实时监测数据
slope_data = {
    'slope_angle': 38.5,
    'slope_height': 60,
    'soil_cohesion': 22.1,
    'internal_friction_angle': 30.5,
    'water_table_depth': 3.2,
    'rainfall_intensity': 7.5,
    'vegetation_coverage': 45,
}

# 预警
alert_system = HazardAlertSystem()
risk_level, probability = alert_system.predict_risk(slope_data)
print(f"风险等级: {['安全', '注意', '警告', '危险'][risk_level]}")
print(f"可信度: {probability:.2%}")
```

---

## 3. 岩爆预测数据 (Rockburst Prediction)

### 输入特征 (Mining Parameters)

| 特征名 | 数据类型 | 单位 | 说明 |
|---|---|---|---|
| depth | float | m | 开挖深度 |
| stress_concentration | float | - | 应力集中系数 |
| uniaxial_strength | float | MPa | 单轴抗压强度 |
| brittleness_index | float | - | 脆性指数 |
| elastic_modulus | float | GPa | 弹性模量 |
| mining_rate | float | m/day | 开挖进度 |
| seismic_velocity | float | m/s | 地震波速度 |
| rock_type | int | 1-5 | 岩石类型编码 |

### 目标变量 (Rockburst Risk Class)

```
岩爆风险等级（四分类）:
- 0: 无岩爆风险 (No Risk)
- 1: 轻微岩爆 (Minor)
- 2: 中等岩爆 (Moderate)
- 3: 强烈岩爆 (Severe)
```

### 数据格式示例

**CSV 格式:**
```csv
event_id,date,depth,stress_concentration,uniaxial_strength,brittleness_index,elastic_modulus,mining_rate,seismic_velocity,rock_type,rockburst_class
E001,2024-01-10,800,2.5,150.2,1.2,45.3,2.1,5500,1,0
E002,2024-01-11,950,3.8,128.5,1.8,38.2,2.5,5200,2,2
E003,2024-01-12,1100,4.2,115.3,2.1,35.1,1.8,4800,3,3
...
```

**使用示例:**
```python
from rockburst_prediction import RockburstAssessment
import pandas as pd

# 加载数据
df = pd.read_csv('mining_data.csv')

# 创建评估器
assessment = RockburstAssessment()

# 预测
X = df.drop('rockburst_class', axis=1)
predictions = assessment.predict(X)

# 输出结果
risk_classes = ['无风险', '轻微', '中等', '强烈']
for i, pred in enumerate(predictions):
    print(f"样本 {i}: {risk_classes[pred]}")
```

---

## 4. 岩心图像识别数据 (Core Image Analysis)

### 支持的图像格式

| 格式 | 扩展名 | 推荐 | 说明 |
|---|---|---|---|
| JPEG | .jpg, .jpeg | ✓ | 常用格式，文件小 |
| PNG | .png | ✓ | 无损压缩，高质量 |
| TIFF | .tif, .tiff | ✓ | 高分辨率专业用途 |
| BMP | .bmp | - | 不推荐，文件大 |

### 图像要求

```
分辨率:
- 最小: 512×512 像素
- 推荐: 1024×1024 或更高
- 高分辨率: 2048×2048 或更高（用于详细分析）

颜色空间:
- RGB 彩色: 用于岩性分类
- 灰度: 用于裂纹检测

质量:
- 清晰度: 高对比度，清晰边界
- 光照: 均匀光照，避免阴影
- 噪声: 低噪声或无噪声
```

### 裂纹检测数据格式

**目录结构:**
```
core_images/
├── training_data/
│   ├── images/
│   │   ├── sample_001.png
│   │   ├── sample_002.png
│   │   └── ...
│   └── masks/
│       ├── sample_001_mask.png
│       ├── sample_002_mask.png
│       └── ...
└── test_data/
    ├── sample_101.png
    ├── sample_102.png
    └── ...
```

**掩膜格式 (Mask):**
- 二值图像 (0-255)
- 白色 (255): 裂纹区域
- 黑色 (0): 非裂纹区域

### 岩性分类数据格式

**目录结构:**
```
rock_samples/
├── granite/
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── ...
├── basalt/
│   ├── sample_001.jpg
│   └── ...
├── limestone/
│   └── ...
└── sandstone/
    └── ...
```

### 使用示例

**裂纹检测:**
```python
from core_image_analysis import CoreImageAnalyzer
import cv2

analyzer = CoreImageAnalyzer()

# 加载图像
image = cv2.imread('core_sample.png')

# 检测裂纹
crack_mask, crack_area = analyzer.detect_cracks(image)

# 保存结果
cv2.imwrite('crack_mask.png', crack_mask)
print(f"裂纹面积占比: {crack_area:.2%}")
```

**岩性分类:**
```python
# 分类岩石类型
rock_type, confidence = analyzer.classify_rock(image)

rock_names = {0: '花岗岩', 1: '玄武岩', 2: '石灰岩', 3: '砂岩'}
print(f"岩石类型: {rock_names[rock_type]}")
print(f"置信度: {confidence:.2%}")
```

---

## 数据准备建议

### 1. 数据清洗

```python
import pandas as pd
import numpy as np

# 移除缺失值
df = df.dropna()

# 处理异常值
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# 数据类型转换
df['weathering_degree'] = df['weathering_degree'].astype(int)
```

### 2. 数据标准化

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化 (z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 或 0-1 范围缩放
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. 训练集/测试集分割

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 常见问题

**Q: 如何处理不同单位的数据？**  
A: 使用标准化或归一化确保特征在同一数量级。

**Q: 图像分辨率不符合要求怎么办？**  
A: 使用 OpenCV 或 PIL 进行图像缩放或插值。

**Q: 缺失数据太多如何处理？**  
A: 可删除该样本、使用插值法或删除缺失特征列。

---

更多问题请参考 [使用指南](usage_guide.md) 或提交 Issue！
