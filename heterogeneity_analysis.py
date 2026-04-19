# heterogeneity_analysis.py

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 读取数据
print("Loading data...")
df_raw = pd.read_excel('data.xlsx')
df_eff = pd.read_excel('carbon_efficiency_results.xlsx')

# 合并数据
df = pd.merge(
    df_raw,
    df_eff[['省份', '城市', '年份', '碳排放效率']],
    on=['省份', '城市', '年份'],
    how='inner'
)

print(f"Merged observations: {len(df)}")

# 因变量
df['eff'] = df['碳排放效率']

# 人均GDP对数及中心化
df['ln_pgdp'] = np.log(df['人均地区生产总值(元)'])
df['ln_pgdp_c'] = df['ln_pgdp'] - df['ln_pgdp'].mean()
df['ln_pgdp_sq_c'] = df['ln_pgdp_c'] ** 2

# 控制变量
df['industry'] = df['第二产业增加值占GDP比重(%)'] / 100
df['ln_fiscal'] = np.log1p(df['地方财政一般预算内支出(万元)'])
df['ln_science'] = np.log1p(df['科学支出(万元)'])
df['ln_so2'] = np.log1p(df['工业二氧化硫排放量(吨)'])

# 分组变量英文映射
df['region_en'] = df['所属地域'].str.strip().map({
    '东部': 'east',
    '中部': 'central',
    '西部': 'west'
})

df['hu_en'] = df['胡焕庸线'].str.strip().map({
    '东南侧': 'southeast',
    '西北侧': 'northwest'
})

# 共线性检查
corr = np.corrcoef(df['ln_pgdp_c'], df['ln_pgdp_sq_c'])[0, 1]
print(f"Centered quadratic correlation: {corr:.6f}")

# 分组回归
def run_regression(data, name):
    if len(data) < 50:
        print(f"Skip group: {name}")
        return None

    data = data.set_index(['城市', '年份'])
    y = data['eff']
    X = data[['ln_pgdp_c', 'ln_pgdp_sq_c', 'industry',
              'ln_fiscal', 'ln_science', 'ln_so2']]

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = model.fit(cov_type='clustered', cluster_entity=True)

    print(f"Regression done: {name} (n={res.nobs})")
    return res

print("\n=== Group regressions ===")
results_region = {g: run_regression(df[df['region_en'] == g], g)
                  for g in df['region_en'].dropna().unique()}

results_hu = {g: run_regression(df[df['hu_en'] == g], g)
              for g in df['hu_en'].dropna().unique()}

# 系数对比表
def extract_coef_table(results_dict, group_label):
    # 生成系数对比表并导出Excel
    core_vars = ['ln_pgdp_c', 'ln_pgdp_sq_c', 'industry', 'ln_fiscal', 'ln_science', 'ln_so2']
    rows = []

    for name, res in results_dict.items():
        if res is None:
            continue

        row = {'Group': name, 'Obs': res.nobs}

        for var in core_vars:
            if var in res.params:
                beta = res.params[var]
                pval = res.pvalues[var]
                stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                row[var] = f"{beta:.4f}{stars}"
            else:
                row[var] = '—'

        rows.append(row)

    df_table = pd.DataFrame(rows)
    df_table.to_excel(f'heterogeneity_{group_label}_table.xlsx', index=False, engine='openpyxl')
    return df_table

if results_region:
    region_table = extract_coef_table(results_region, 'region')
    print("Region coefficient table saved to heterogeneity_region_table.xlsx")
    print(region_table.to_string(index=False))

if results_hu:
    hu_table = extract_coef_table(results_hu, 'hu')
    print("Hu line coefficient table saved to heterogeneity_hu_table.xlsx")
    print(hu_table.to_string(index=False))

# Z检验
print("\n=== Z test ===")

z_results = []

def z_test(res1, res2, var):
    b1, b2 = res1.params[var], res2.params[var]
    se1, se2 = res1.std_errors[var], res2.std_errors[var]
    z = (b1 - b2) / np.sqrt(se1**2 + se2**2)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

if 'east' in results_region and 'west' in results_region:
    for var in ['ln_pgdp_c', 'ln_pgdp_sq_c']:
        z, p = z_test(results_region['east'], results_region['west'], var)
        z_results.append({'Comparison': 'East vs West', 'Variable': var, 'Z': z, 'P-value': p})
        print(f"{var}: z={z:.3f}, p={p:.3f}")

# Wald检验
print("\n=== Wald test ===")

wald_results = []

def wald_test_matrix(df, group_col, ref, output_name):
    df = df.copy()
    groups = df[group_col].dropna().unique()
    others = [g for g in groups if g != ref]

    base_vars = ['ln_pgdp_c', 'ln_pgdp_sq_c', 'industry',
                 'ln_fiscal', 'ln_science', 'ln_so2']

    # 构造交互项
    for g in others:
        df[f'd_{g}'] = (df[group_col] == g).astype(int)
        for v in base_vars:
            df[f'{v}_x_{g}'] = df[v] * df[f'd_{g}']

    X_vars = base_vars + [f'{v}_x_{g}' for g in others for v in base_vars]

    data = df.set_index(['城市', '年份'])
    y = data['eff']
    X = data[X_vars]

    model = PanelOLS(
        y, X,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )

    res = model.fit(cov_type='clustered', cluster_entity=True)

    param_names = res.params.index.tolist()
    test_vars = [v for v in param_names if '_x_' in v]

    if len(test_vars) == 0:
        print(f"No interaction terms in {group_col}")
        return None

    k = len(param_names)
    q = len(test_vars)

    R = np.zeros((q, k))

    for i, var in enumerate(test_vars):
        idx = param_names.index(var)
        R[i, idx] = 1

    r = np.zeros(q)

    wald = res.wald_test(R, r)

    print(f"[{output_name}] Wald test result:")
    print(f"stat = {wald.stat:.4f}")
    print(f"p-value = {wald.pval:.4f}")

    if wald.pval < 0.05:
        conclusion = "Significant heterogeneity detected"
    else:
        conclusion = "No significant difference found"

    print(f"Conclusion: {conclusion}")

    wald_results.append({
        'Group': output_name,
        'Wald Stat': wald.stat,
        'P-value': wald.pval,
        'Conclusion': conclusion
    })

    with open(f'wald_interaction_{output_name}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Interaction regression results - {output_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(res.summary.as_text())
        f.write("\n\n" + "=" * 70 + "\n")
        f.write(f"Wald test: stat={wald.stat:.4f}, p={wald.pval:.4f}\n")
        f.write(f"Conclusion: {conclusion}\n")

    return wald

wald_region = wald_test_matrix(df, 'region_en', 'central', 'region')
wald_hu = wald_test_matrix(df, 'hu_en', 'southeast', 'hu_line')

# 汇总输出
with open('heterogeneity_summary.txt', 'w', encoding='utf-8') as f:
    f.write("Heterogeneity analysis summary\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Sample size: {len(df)}\n")
    f.write(f"Centered quadratic correlation: {corr:.6f}\n\n")

    f.write("Group regression sample sizes\n")
    f.write("-" * 40 + "\n")

    for g, res in results_region.items():
        if res: f.write(f"{g}: {res.nobs}\n")
    for g, res in results_hu.items():
        if res: f.write(f"{g}: {res.nobs}\n")

    f.write("\nCoefficient tables saved as Excel files\n")

    if results_region:
        f.write(region_table.to_string(index=False))
        f.write("\n")

    if results_hu:
        f.write(hu_table.to_string(index=False))
        f.write("\n")

    f.write("\nZ tests (East vs West)\n")
    for zr in z_results:
        f.write(f"{zr['Variable']}: z={zr['Z']:.3f}, p={zr['P-value']:.3f}\n")

    f.write("\nWald tests\n")
    for wr in wald_results:
        f.write(f"{wr['Group']}: stat={wr['Wald Stat']:.4f}, p={wr['P-value']:.4f} -> {wr['Conclusion']}\n")

# 分组回归完整输出
with open('heterogeneity_results_region.txt', 'w', encoding='utf-8') as f:
    f.write("Region group regression results\n")
    f.write("=" * 70 + "\n\n")
    for g, res in results_region.items():
        if res:
            f.write(f"Group: {g} (n={res.nobs})\n")
            f.write(res.summary.as_text())
            f.write("\n" + "-" * 70 + "\n\n")

with open('heterogeneity_results_hu.txt', 'w', encoding='utf-8') as f:
    f.write("Hu line group regression results\n")
    f.write("=" * 70 + "\n\n")
    for g, res in results_hu.items():
        if res:
            f.write(f"Group: {g} (n={res.nobs})\n")
            f.write(res.summary.as_text())
            f.write("\n" + "-" * 70 + "\n\n")

print("\nDone. Output files generated:")
print("  heterogeneity_region_table.xlsx")
print("  heterogeneity_hu_table.xlsx")
print("  heterogeneity_summary.txt")
print("  heterogeneity_results_region.txt")
print("  heterogeneity_results_hu.txt")
print("  wald_interaction_region.txt")
print("  wald_interaction_hu_line.txt")