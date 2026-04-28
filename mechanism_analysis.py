# mechanism_analysis.py

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# 数据读取
df_raw = pd.read_excel('data.xlsx')
df_eff = pd.read_excel('carbon_efficiency_results.xlsx')

df = pd.merge(
    df_raw,
    df_eff[['省份', '城市', '年份', '碳排放效率']],
    on=['省份', '城市', '年份'],
    how='inner'
)

print(f"Sample size: {len(df)}")

# 变量构造
df['eff'] = df['碳排放效率']

df['ln_pgdp'] = np.log(df['人均地区生产总值(元)'].clip(lower=1))
mean_ln = df['ln_pgdp'].mean()
df['ln_pgdp_c'] = df['ln_pgdp'] - mean_ln
df['ln_pgdp_sq_c'] = df['ln_pgdp_c'] ** 2

df['industry'] = df['第二产业增加值占GDP比重(%)'] / 100
df['ln_fiscal'] = np.log1p(df['地方财政一般预算内支出(万元)'])

df['ln_so2'] = np.log1p(df['工业二氧化硫排放量(吨)'])

df = df.set_index(['城市', '年份'])

# 回归函数
def run_fe(y, X):
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = model.fit(cov_type='clustered', cluster_entity=True)
    return res

# Sobel检验
def sobel_test(a, b, se_a, se_b):
    z = (a * b) / np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p

# 中介分析函数
def mediation_ekc(mech_name, mediator_col, control_cols):
    print("\n" + "=" * 60)
    print(f"Mechanism: {mech_name} (Mediator: {mediator_col})")
    print(f"Controls: {control_cols}")
    print("=" * 60)

    core_x = ['ln_pgdp_c', 'ln_pgdp_sq_c']

    X1 = df[core_x + control_cols]
    y1 = df['eff']
    res1 = run_fe(y1, X1)

    X2 = df[core_x + control_cols]
    y2 = df[mediator_col]
    res2 = run_fe(y2, X2)

    X3 = df[core_x + control_cols + [mediator_col]]
    y3 = df['eff']
    res3 = run_fe(y3, X3)

    total_linear = res1.params['ln_pgdp_c']
    total_sq = res1.params['ln_pgdp_sq_c']

    a_linear = res2.params['ln_pgdp_c']
    a_sq = res2.params['ln_pgdp_sq_c']
    se_a_linear = res2.std_errors['ln_pgdp_c']
    se_a_sq = res2.std_errors['ln_pgdp_sq_c']

    b = res3.params[mediator_col]
    se_b = res3.std_errors[mediator_col]

    direct_linear = res3.params['ln_pgdp_c']
    direct_sq = res3.params['ln_pgdp_sq_c']

    indirect_linear = a_linear * b
    indirect_sq = a_sq * b

    ratio_linear = indirect_linear / total_linear if abs(total_linear) > 1e-6 else np.nan
    ratio_sq = indirect_sq / total_sq if abs(total_sq) > 1e-6 else np.nan

    z_lin, p_lin = sobel_test(a_linear, b, se_a_linear, se_b)
    z_sq, p_sq = sobel_test(a_sq, b, se_a_sq, se_b)

    print(f"Total effects: linear = {total_linear:.4f}, squared = {total_sq:.4f}")
    print(f"Path a: linear = {a_linear:.4f}, squared = {a_sq:.4f}")
    print(f"Path b: {b:.4f} (p={res3.pvalues[mediator_col]:.4f})")
    print(f"Direct effects: linear = {direct_linear:.4f}, squared = {direct_sq:.4f}")
    print(f"Indirect effects: linear = {indirect_linear:.4f} (ratio={ratio_linear:.3%}), "
          f"squared = {indirect_sq:.4f} (ratio={ratio_sq:.3%})")
    print(f"Sobel linear: z={z_lin:.3f}, p={p_lin:.4f}")
    print(f"Sobel squared: z={z_sq:.3f}, p={p_sq:.4f}")

    if p_lin < 0.05:
        type_lin = "Partial mediation" if abs(direct_linear) < abs(total_linear) else "Full/Suppression"
    else:
        type_lin = "No mediation"

    if p_sq < 0.05:
        type_sq = "Partial mediation" if abs(direct_sq) < abs(total_sq) else "Full/Suppression"
    else:
        type_sq = "No mediation"

    print(f"Mediation type linear: {type_lin}, squared: {type_sq}")

    return {
        'Mechanism': mech_name,
        'Mediator': mediator_col,
        'Total_linear': total_linear,
        'Total_squared': total_sq,
        'a_linear': a_linear,
        'a_squared': a_sq,
        'b': b,
        'Direct_linear': direct_linear,
        'Direct_squared': direct_sq,
        'Indirect_linear': indirect_linear,
        'Indirect_squared': indirect_sq,
        'Ratio_linear': ratio_linear,
        'Ratio_squared': ratio_sq,
        'Sobel_Z_linear': z_lin,
        'Sobel_p_linear': p_lin,
        'Sobel_Z_squared': z_sq,
        'Sobel_p_squared': p_sq,
        'Med_Type_linear': type_lin,
        'Med_Type_squared': type_sq
    }

results = []

pollution_ctrl = ['industry', 'ln_fiscal']
results.append(mediation_ekc('Pollution Mechanism', 'ln_so2', pollution_ctrl))

df_out = pd.DataFrame(results)
df_out.to_excel('mechanism_analysis_results.xlsx', index=False)

print("\n" + "=" * 70)
print("Mechanism analysis completed.")
print(df_out.to_string(index=False))