# twfe_stirpat_regression.py
"""
双向固定效应面板回归 (STIRPAT framework)
  被解释变量: 碳排放效率 (SBM测算)
  估计方法: PanelOLS (双向固定效应) + 城市聚类稳健标准误
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')


# 读取数据
df_raw = pd.read_excel('data.xlsx')
df_eff = pd.read_excel('carbon_efficiency_results.xlsx')

df = pd.merge(
    df_raw,
    df_eff[['省份', '城市', '年份', '碳排放效率']],
    on=['省份', '城市', '年份'],
    how='inner'
)

print(f"Data loaded. Observations: {len(df)}")


# 被解释变量
df['eff'] = df['碳排放效率']

# 核心解释变量：人均GDP (log + demeaned)
df['ln_pgdp'] = np.log(df['人均地区生产总值(元)'].clip(lower=1))
mean_ln_pgdp = df['ln_pgdp'].mean()
df['ln_pgdp_c'] = df['ln_pgdp'] - mean_ln_pgdp
df['ln_pgdp_sq_c'] = df['ln_pgdp_c'] ** 2

# 控制变量
df['industry'] = df['第二产业增加值占GDP比重(%)'] / 100
df['ln_fiscal'] = np.log(df['地方财政一般预算内支出(万元)'].clip(lower=1))
df['ln_science'] = np.log(df['科学支出(万元)'].clip(lower=1))
df['ln_so2'] = np.log1p(df['工业二氧化硫排放量(吨)'])

# 设置面板索引
df = df.set_index(['城市', '年份'])

y = df['eff']
X = df[['ln_pgdp_c', 'ln_pgdp_sq_c', 'industry',
        'ln_fiscal', 'ln_science', 'ln_so2']]


# 双向固定效应回归
print("Running two-way fixed effects model...")

model = PanelOLS(y, X, entity_effects=True, time_effects=True)
result = model.fit(cov_type='clustered', cluster_entity=True)

print("Regression completed.")


# EKC拐点计算
beta1 = result.params['ln_pgdp_c']
beta2 = result.params['ln_pgdp_sq_c']

turning_point_ln = mean_ln_pgdp - beta1 / (2 * beta2)
turning_point_gdp = np.exp(turning_point_ln)

print(f"EKC turning point (GDP per capita): {turning_point_gdp:.2f}")


# 多重共线性 (within-demeaned VIF)
df_demean = df.copy()

for col in X.columns:
    df_demean[col] = df_demean[col] - df_demean.groupby(level='城市')[col].transform('mean')

X_demean = sm.add_constant(df_demean[X.columns])

vif_data = pd.DataFrame({
    'Variable': X_demean.columns,
    'VIF': [
        variance_inflation_factor(X_demean.values, i)
        for i in range(X_demean.shape[1])
    ]
})

print("VIF calculation completed.")


# 保存结果
with open('twfe_regression_results.txt', 'w', encoding='utf-8') as f:
    f.write("Two-way Fixed Effects Regression Results\n")
    f.write("=" * 70 + "\n\n")
    f.write(result.summary.as_text())
    f.write("\n\nEKC Turning Point (GDP per capita): {:.2f}\n".format(turning_point_gdp))
    f.write("\nVIF (within-demeaned):\n")
    f.write(vif_data.to_string(index=False))

print("Results saved to twfe_regression_results.txt")