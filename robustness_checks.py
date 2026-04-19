# robustness_checks.py
"""
稳健性检验
统一EKC拐点公式(基于原始模型空间)
常数项被固定效应吸收
新增EKC有效性标志(beta1>0, beta2<0, 拐点在样本内)
保留稳健性处理(winsorize, 变量映射, 聚类说明)
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from scipy.stats.mstats import winsorize
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("Robustness Check")
print("Note: City and year fixed effects included, constant absorbed.")
print("=" * 70)

# 读取数据
df_raw = pd.read_excel('data.xlsx')
df_eff = pd.read_excel('carbon_efficiency_results.xlsx')

df = pd.merge(
    df_raw,
    df_eff[['省份', '城市', '年份', '碳排放效率']],
    on=['省份', '城市', '年份'],
    how='inner'
)

print(f"Sample size after merge: {len(df)}")

df['eff'] = df['碳排放效率']
df['pgdp'] = df['人均地区生产总值(元)']
df['ln_pgdp'] = np.log(df['pgdp'])

mean_ln_pgdp = df['ln_pgdp'].mean()
df['ln_pgdp_c'] = df['ln_pgdp'] - mean_ln_pgdp
df['ln_pgdp_sq_c'] = df['ln_pgdp_c'] ** 2

df['industry'] = df['第二产业增加值占GDP比重(%)'] / 100
df['ln_fiscal'] = np.log1p(df['地方财政一般预算内支出(万元)'])
df['ln_science'] = np.log1p(df['科学支出(万元)'])
df['ln_so2'] = np.log1p(df['工业二氧化硫排放量(吨)'])

# 稳健性变量
df['co2_intensity'] = df['CO2排放总量(吨)'] / df['地区生产总值(万元)']
df['ln_co2_int'] = np.log1p(df['co2_intensity'])

df['pgdp_raw'] = df['pgdp'] / 10000
df['pgdp_raw_c'] = df['pgdp_raw'] - df['pgdp_raw'].mean()
df['pgdp_raw_sq_c'] = df['pgdp_raw_c'] ** 2

# 缩尾处理
var_list = [
    'eff', 'ln_pgdp_c', 'ln_pgdp_sq_c', 'industry',
    'ln_fiscal', 'ln_science', 'ln_so2'
]

for var in var_list:
    df[var + '_w'] = winsorize(df[var], limits=(0.01, 0.01)).data

df_no_muni = df[~df['城市'].isin(['北京市', '天津市', '上海市', '重庆市'])].copy()

# 变量映射
var_map = {
    'baseline': {
        'x': 'ln_pgdp_c',
        'x2': 'ln_pgdp_sq_c',
        'y': 'eff',
        'ctrl': ['industry', 'ln_fiscal', 'ln_science', 'ln_so2'],
        'is_log': True,
        'mean_aux': mean_ln_pgdp
    },
    'y_co2': {
        'x': 'ln_pgdp_c',
        'x2': 'ln_pgdp_sq_c',
        'y': 'ln_co2_int',
        'ctrl': ['industry', 'ln_fiscal', 'ln_science', 'ln_so2'],
        'is_log': True,
        'mean_aux': mean_ln_pgdp
    },
    'x_raw': {
        'x': 'pgdp_raw_c',
        'x2': 'pgdp_raw_sq_c',
        'y': 'eff',
        'ctrl': ['industry', 'ln_fiscal', 'ln_science', 'ln_so2'],
        'is_log': False,
        'mean_aux': df['pgdp_raw'].mean()
    },
    'winsor': {
        'x': 'ln_pgdp_c_w',
        'x2': 'ln_pgdp_sq_c_w',
        'y': 'eff_w',
        'ctrl': ['industry_w', 'ln_fiscal_w', 'ln_science_w', 'ln_so2_w'],
        'is_log': True,
        'mean_aux': mean_ln_pgdp
    },
    'no_muni': {
        'x': 'ln_pgdp_c',
        'x2': 'ln_pgdp_sq_c',
        'y': 'eff',
        'ctrl': ['industry', 'ln_fiscal', 'ln_science', 'ln_so2'],
        'is_log': True,
        'mean_aux': mean_ln_pgdp
    }
}

# 回归函数
def run_model(data, model_key, cluster_type='entity'):
    """
    双向固定效应模型
    cluster_type entity city聚类 time年份聚类 both双向聚类
    """
    spec = var_map[model_key]
    data = data.set_index(['城市', '年份'])

    y = data[spec['y']]
    X = data[[spec['x'], spec['x2']] + spec['ctrl']]

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)

    if cluster_type == 'entity':
        res = model.fit(cov_type='clustered', cluster_entity=True)
    elif cluster_type == 'time':
        res = model.fit(cov_type='clustered', cluster_time=True)
    elif cluster_type == 'both':
        res = model.fit(
            cov_type='clustered',
            cluster_entity=True,
            cluster_time=True
        )
    else:
        raise ValueError("cluster_type must be entity, time or both")

    return res


def compute_ekc(res, model_key, df_original):
    """
    EKC拐点计算
    返回(拐点, 形状, 是否有效)
    """
    spec = var_map[model_key]

    beta1 = res.params.get(spec['x'], np.nan)
    beta2 = res.params.get(spec['x2'], np.nan)

    if np.isnan(beta1) or np.isnan(beta2) or abs(beta2) < 1e-6:
        return np.nan, "no turning point", False

    if spec['is_log']:
        turning_point = np.exp(-beta1 / (2 * beta2) + spec['mean_aux'])
    else:
        turning_point = (-beta1 / (2 * beta2)) * 10000

    if beta1 > 0 and beta2 < 0:
        shape = "inverted U"
    elif beta1 < 0 and beta2 > 0:
        shape = "U shape"
    elif beta1 > 0 and beta2 > 0:
        shape = "monotonic increase"
    elif beta1 < 0 and beta2 < 0:
        shape = "monotonic decrease"
    else:
        shape = "undetermined"

    pgdp_min = df_original['pgdp'].min()
    pgdp_max = df_original['pgdp'].max()

    in_sample = (pgdp_min <= turning_point <= pgdp_max)

    if not in_sample:
        shape += "(out of sample)"

    ekc_valid = (beta1 > 0) and (beta2 < 0) and in_sample

    return turning_point, shape, ekc_valid


# 稳健性检验
print("\nRunning robustness checks...")

results = {}
ekc_records = []

# 1 baseline
print("Running baseline model...")
res_base = run_model(df, 'baseline', 'entity')
results['Baseline'] = res_base

tp, shape, valid = compute_ekc(res_base, 'baseline', df)
ekc_records.append({
    'Model': 'Baseline',
    'beta1_sign': '+' if res_base.params['ln_pgdp_c'] > 0 else '-',
    'beta2_sign': '+' if res_base.params['ln_pgdp_sq_c'] > 0 else '-',
    'turning_point': tp,
    'shape': shape,
    'EKC_valid': valid
})

# 2 CO2 intensity
print("Running CO2 intensity model...")
res_y = run_model(df, 'y_co2', 'entity')
results['CO2 Intensity'] = res_y

tp, shape, valid = compute_ekc(res_y, 'y_co2', df)
ekc_records.append({
    'Model': 'CO2 Intensity',
    'beta1_sign': '+' if res_y.params['ln_pgdp_c'] > 0 else '-',
    'beta2_sign': '+' if res_y.params['ln_pgdp_sq_c'] > 0 else '-',
    'turning_point': tp,
    'shape': shape,
    'EKC_valid': valid
})

# 3 raw GDP
print("Running raw GDP model...")
res_x = run_model(df, 'x_raw', 'entity')
results['Raw GDP'] = res_x

tp, shape, valid = compute_ekc(res_x, 'x_raw', df)
ekc_records.append({
    'Model': 'Raw GDP',
    'beta1_sign': '+' if res_x.params['pgdp_raw_c'] > 0 else '-',
    'beta2_sign': '+' if res_x.params['pgdp_raw_sq_c'] > 0 else '-',
    'turning_point': tp,
    'shape': shape,
    'EKC_valid': valid
})

# 4 winsorized
print("Running winsorized model...")
res_w = run_model(df, 'winsor', 'entity')
results['Winsorized'] = res_w

tp, shape, valid = compute_ekc(res_w, 'winsor', df)
ekc_records.append({
    'Model': 'Winsorized',
    'beta1_sign': '+' if res_w.params['ln_pgdp_c_w'] > 0 else '-',
    'beta2_sign': '+' if res_w.params['ln_pgdp_sq_c_w'] > 0 else '-',
    'turning_point': tp,
    'shape': shape,
    'EKC_valid': valid
})

# 5 no municipalities
print("Running no municipalities model...")
res_muni = run_model(df_no_muni, 'no_muni', 'entity')
results['No Municipalities'] = res_muni

tp, shape, valid = compute_ekc(res_muni, 'no_muni', df_no_muni)
ekc_records.append({
    'Model': 'No Municipalities',
    'beta1_sign': '+' if res_muni.params['ln_pgdp_c'] > 0 else '-',
    'beta2_sign': '+' if res_muni.params['ln_pgdp_sq_c'] > 0 else '-',
    'turning_point': tp,
    'shape': shape,
    'EKC_valid': valid
})

# 6 two-way clustering
print("Running two-way clustering model...")
res_both = run_model(df, 'baseline', 'both')
results['Two-way Clustering'] = res_both

tp, shape, valid = compute_ekc(res_both, 'baseline', df)
ekc_records.append({
    'Model': 'Two-way Clustering',
    'beta1_sign': '+' if res_both.params['ln_pgdp_c'] > 0 else '-',
    'beta2_sign': '+' if res_both.params['ln_pgdp_sq_c'] > 0 else '-',
    'turning_point': tp,
    'shape': shape,
    'EKC_valid': valid
})

# 系数表
def extract_row(name, res, model_key):
    spec = var_map[model_key]
    row = {'Model': name, 'Obs': res.nobs}

    x = spec['x']
    x2 = spec['x2']

    for var, label in [(x, 'linear'), (x2, 'quadratic')]:
        beta = res.params.get(var, np.nan)
        pval = res.pvalues.get(var, np.nan)

        stars = ''
        if pval < 0.01:
            stars = '***'
        elif pval < 0.05:
            stars = '**'
        elif pval < 0.1:
            stars = '*'

        row[label] = f"{beta:.4f}{stars}" if not np.isnan(beta) else 'NA'

    return row


coef_rows = []
coef_rows.append(extract_row('Baseline', results['Baseline'], 'baseline'))
coef_rows.append(extract_row('CO2 Intensity', results['CO2 Intensity'], 'y_co2'))
coef_rows.append(extract_row('Raw GDP', results['Raw GDP'], 'x_raw'))
coef_rows.append(extract_row('Winsorized', results['Winsorized'], 'winsor'))
coef_rows.append(extract_row('No Municipalities', results['No Municipalities'], 'no_muni'))
coef_rows.append(extract_row('Two-way Clustering', results['Two-way Clustering'], 'baseline'))

coef_df = pd.DataFrame(coef_rows)

print("\nCoefficient table:")
print(coef_df.to_string(index=False))

coef_df.to_excel('robustness_coefficients.xlsx', index=False)

# EKC表
ekc_df = pd.DataFrame(ekc_records)

ekc_df['Turning Point'] = ekc_df['turning_point'].apply(
    lambda x: f"{x:,.0f}" if pd.notna(x) else "NA"
)

ekc_df = ekc_df[['Model', 'beta1_sign', 'beta2_sign', 'Turning Point', 'shape', 'EKC_valid']]

print("\nEKC validity table:")
print(ekc_df.to_string(index=False))

ekc_df.to_excel('robustness_ekc_validity.xlsx', index=False)

# 保存结果
with open('robustness_full_results.txt', 'w', encoding='utf-8') as f:
    f.write("Robustness check full results\n")
    f.write("City and year fixed effects included\n")
    f.write("=" * 70 + "\n\n")

    for name, res in results.items():
        f.write(f"Model: {name} (Obs: {res.nobs})\n")
        f.write(res.summary.as_text())
        f.write("\n\n" + "-" * 70 + "\n\n")

print("\nCompleted")
print("Files saved:")
print("robustness_coefficients.xlsx")
print("robustness_ekc_validity.xlsx")
print("robustness_full_results.txt")