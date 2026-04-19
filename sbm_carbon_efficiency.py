# sbm_carbon_efficiency.py

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def standard_sbm_undesirable_vrs_cc(X, Y_g, Y_b):
    """
    SBM模型(含非期望产出, VRS假设, Charnes-Cooper变换)

    参数:
        X   : 投入矩阵(n x m)
        Y_g : 期望产出矩阵(n x s1)
        Y_b : 非期望产出矩阵(n x s2)

    返回:
        eff : 各DMU效率值, 范围(0, 1]
    """

    n, m = X.shape
    s1 = Y_g.shape[1]
    s2 = Y_b.shape[1]

    eff = np.full(n, np.nan)

    # 逐个DMU求解线性规划
    for i in tqdm(range(n), desc='SBM solving'):

        x0, yg0, yb0 = X[i], Y_g[i], Y_b[i]

        # 决策变量: lambda_tilde(n) + s-(m) + s+(s1) + s-(s2) + t(1)
        n_vars = n + m + s1 + s2 + 1
        t_idx = n_vars - 1

        c = np.zeros(n_vars)

        # 目标函数: min t - (1/m) * sum(s-_k / x0_k)
        c[t_idx] = 1.0
        for k in range(m):
            if x0[k] > 1e-8:
                c[n + k] = -1.0 / (m * x0[k])

        A_eq, b_eq = [], []

        # 投入约束: sum_j(lambda_tilde_j * x_kj) + s-_k = t * x0_k
        for k in range(m):
            row = np.zeros(n_vars)
            row[:n] = X[:, k]
            row[n + k] = 1.0
            row[t_idx] = -x0[k]
            A_eq.append(row)
            b_eq.append(0.0)

        # 期望产出约束: sum_j(lambda_tilde_j * yg_rj) - s+_r = t * yg0_r
        for r in range(s1):
            row = np.zeros(n_vars)
            row[:n] = Y_g[:, r]
            row[n + m + r] = -1.0
            row[t_idx] = -yg0[r]
            A_eq.append(row)
            b_eq.append(0.0)

        # 非期望产出约束: sum_j(lambda_tilde_j * yb_tj) + s-_t = t * yb0_t
        for t in range(s2):
            row = np.zeros(n_vars)
            row[:n] = Y_b[:, t]
            row[n + m + s1 + t] = 1.0
            row[t_idx] = -yb0[t]
            A_eq.append(row)
            b_eq.append(0.0)

        # 归一化约束: t + (1/(s1+s2)) * [sum(s+_r / yg0_r) + sum(s-_t / yb0_t)] = 1
        row_norm = np.zeros(n_vars)
        row_norm[t_idx] = 1.0
        denom = s1 + s2
        for r in range(s1):
            if yg0[r] > 1e-8:
                row_norm[n + m + r] = 1.0 / (denom * yg0[r])
        for t in range(s2):
            if yb0[t] > 1e-8:
                row_norm[n + m + s1 + t] = 1.0 / (denom * yb0[t])
        A_eq.append(row_norm)
        b_eq.append(1.0)

        # VRS约束: sum_j(lambda_tilde_j) = t
        row_vrs = np.zeros(n_vars)
        row_vrs[:n] = 1.0
        row_vrs[t_idx] = -1.0
        A_eq.append(row_vrs)
        b_eq.append(0.0)

        # 求解线性规划
        try:
            res = linprog(
                c,
                A_eq=np.array(A_eq),
                b_eq=np.array(b_eq),
                bounds=[(0, None)] * n_vars,
                method='highs'
            )
            if res.success:
                eff[i] = res.fun   # 目标函数最优值即为效率值
        except Exception:
            pass

    return eff


if __name__ == "__main__":

    # 读取数据
    df = pd.read_excel('data.xlsx')

    # 变量选择: 投入, 期望产出, 非期望产出
    cols = [
        '城镇非私营单位从业人员数(万人)',
        '全社会用电量_万千瓦时',
        '地区生产总值(万元)',
        'CO2排放总量(吨)'
    ]

    # 非正值处理: 防止除零错误
    for c in cols:
        df[c] = df[c].clip(lower=1e-6)

    # 构造矩阵
    X_raw = df[cols[:2]].values
    Yg_raw = df[[cols[2]]].values
    Yb_raw = df[[cols[3]]].values

    # 均值标准化: 消除量纲影响
    X = X_raw / X_raw.mean(axis=0)
    Yg = Yg_raw / Yg_raw.mean(axis=0)
    Yb = Yb_raw / Yb_raw.mean(axis=0)

    # SBM效率计算
    eff = standard_sbm_undesirable_vrs_cc(X, Yg, Yb)

    # 结果输出
    df['碳排放效率'] = eff
    df[['省份', '城市', '年份', '碳排放效率']].to_excel(
        'carbon_efficiency_results.xlsx',
        index=False
    )

    # 描述性统计
    print(
        f"valid: {np.sum(~np.isnan(eff))}, "
        f"range: [{np.nanmin(eff):.4f}, {np.nanmax(eff):.4f}], "
        f"mean: {np.nanmean(eff):.4f}, std: {np.nanstd(eff):.4f}"
    )