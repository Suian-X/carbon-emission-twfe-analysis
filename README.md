# 碳排放效率与影响因素分析：基于 STIRPAT 与双向固定效应模型

## 一、项目简介

本项目基于中国城市面板数据，首先利用包含非期望产出的 SBM（Slack-Based Measure）模型测算碳排放效率，在此基础上引入 STIRPAT 理论框架，采用双向固定效应模型（Two-Way Fixed Effects, TWFE）对碳排放效率的影响因素进行系统实证分析，并进一步开展异质性分析与稳健性检验。

研究目标在于识别经济发展水平及相关控制变量对碳排放效率的影响机制，并检验环境库兹涅茨曲线（EKC）假说在样本中的适用性。

---

## 二、数据说明

**数据类型：** 城市层面面板数据

**关键变量包括：**

* 投入指标：劳动力、能源消耗
* 期望产出：地区生产总值
* 非期望产出：CO₂排放量
* 解释变量：人均GDP（对数及其平方项）
* 控制变量：产业结构、财政支出、科技支出、污染排放等

所有变量在建模前均进行了必要的预处理，包括非正值修正、对数变换及中心化处理。

---

## 三、方法框架

### （一）碳排放效率测算

采用 SBM 模型测算碳排放效率，具体特征如下：

* 纳入非期望产出（CO₂排放）
* 假设可变规模报酬（VRS）
* 通过 Charnes-Cooper 变换将分式规划转化为线性规划求解

该部分由 `sbm_carbon_efficiency.py` 实现，输出各决策单元（DMU）的效率值。

---

### （二）基准回归模型

在 STIRPAT 理论框架下，构建如下双向固定效应模型：

```
eff_it = β1 ln(pgdp_it) + β2 [ln(pgdp_it)]² + γX_it + μ_i + λ_t + ε_it
```

其中：

* `eff_it`：碳排放效率
* `ln(pgdp_it)`：人均GDP对数（经中心化处理）
* `X_it`：控制变量向量
* `μ_i`：个体固定效应（城市）
* `λ_t`：时间固定效应（年份）

估计方法采用 PanelOLS，并使用城市层面的聚类稳健标准误。

同时，根据估计结果计算 EKC 拐点，并分析其经济含义。

---

### （三）异质性分析

为考察不同区域特征下的影响差异，开展分组回归分析，具体包括：

* 区域分组：东部、中部、西部
* 胡焕庸线分组：东南侧、西北侧

在此基础上进一步进行：

* 系数对比分析
* Z 检验（组间系数差异）
* 基于交互项的 Wald 检验（整体异质性显著性）

相关结果以文本与表格形式输出。

---

### （四）稳健性检验

为验证结论的稳健性，构建多种替代模型与处理方式，包括：

* 更换被解释变量（碳排放强度）
* 更换核心解释变量形式（非对数人均GDP）
* 对主要变量进行缩尾处理（Winsorize）
* 剔除直辖市样本
* 采用双向聚类稳健标准误

同时对各模型结果进行 EKC 拐点识别，并判定其是否满足“倒U型且位于样本区间内”的有效性条件。

---

## 四、文件结构

```
.
├── data.xlsx                          # 原始数据
├── carbon_efficiency_results.xlsx     # 碳排放效率结果

├── sbm_carbon_efficiency.py           # SBM效率测算
├── twfe_stirpat_regression.py         # 基准回归（TWFE）
├── heterogeneity_analysis.py          # 异质性分析
├── robustness_checks.py               # 稳健性检验

├── twfe_regression_results.txt
├── heterogeneity_*.txt / *.xlsx
├── robustness_*.txt / *.xlsx
```

---

## 五、运行说明

### 1. 安装依赖环境

```bash
pip install pandas numpy scipy statsmodels linearmodels openpyxl tqdm
```

### 2. 按如下顺序执行脚本

```bash
python sbm_carbon_efficiency.py
python twfe_stirpat_regression.py
python heterogeneity_analysis.py
python robustness_checks.py
```

---

## 六、结果说明

* 碳排放效率结果：`carbon_efficiency_results.xlsx`
* 基准回归结果：`twfe_regression_results.txt`
* 异质性分析结果：`heterogeneity_*`
* 稳健性检验结果：`robustness_*`

所有结果均可由代码直接复现。

---

## 七、说明

本项目用于学术研究与方法复现，相关结果仅供参考。如用于进一步研究，请结合具体数据背景进行解释。
