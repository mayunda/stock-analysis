import akshare as ak
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ===== 第1步：拉取数据（这里改用近3年数据，样本量太小统计没有意义）=====
stock_code = "600519"
df = ak.stock_zh_a_hist(
    symbol=stock_code,
    period="daily",
    start_date="20220101",
    end_date="20241231",
    adjust="qfq"
)
df = df.sort_values("日期").reset_index(drop=True)

close_prices = df["收盘"].values

# ===== 第2步：平滑 + 求导（和上一步一样）=====
window_length = 11
polyorder = 3

df["平滑收盘价"] = savgol_filter(close_prices, window_length, polyorder)
df["一阶导"] = savgol_filter(close_prices, window_length, polyorder, deriv=1)
df["二阶导"] = savgol_filter(close_prices, window_length, polyorder, deriv=2)

# ===== 第3步：识别"一阶导由负转正"的信号点 =====
# 用 shift(1) 拿到"前一天"的一阶导值，和"今天"的比较
df["一阶导_昨天"] = df["一阶导"].shift(1)

# 信号条件：昨天是负的，今天变成正的 = 由跌转涨的拐点
df["买入信号"] = (df["一阶导_昨天"] < 0) & (df["一阶导"] > 0)

signal_dates = df[df["买入信号"]].index.tolist()
print(f"一共找到 {len(signal_dates)} 个买入信号点\n")

# ===== 第4步：回测 —— 看每次信号后5天/10天的涨跌情况 =====
results_5d = []
results_10d = []

for idx in signal_dates:
    entry_price = df.loc[idx, "收盘"]

    # 检查后面是否还有5天/10天数据，不够的跳过（避免最后几天数据不全报错）
    if idx + 5 < len(df):
        price_5d_later = df.loc[idx + 5, "收盘"]
        return_5d = (price_5d_later - entry_price) / entry_price * 100
        results_5d.append(return_5d)

    if idx + 10 < len(df):
        price_10d_later = df.loc[idx + 10, "收盘"]
        return_10d = (price_10d_later - entry_price) / entry_price * 100
        results_10d.append(return_10d)

# ===== 第5步：统计结果 =====
results_5d = np.array(results_5d)
results_10d = np.array(results_10d)

print("===== 5天后表现 =====")
print(f"样本数量: {len(results_5d)}")
print(f"平均涨跌幅: {results_5d.mean():.2f}%")
print(f"上涨概率: {(results_5d > 0).mean() * 100:.1f}%")
print(f"最大涨幅: {results_5d.max():.2f}%")
print(f"最大跌幅: {results_5d.min():.2f}%")
print(f"最大跌幅: {results_5d.min():.2f}%")

# ===== 新增：剔除极端值后的检验 =====
# 把5天涨跌幅从大到小排序，去掉最高的3个（极端值），看剩下的表现
sorted_5d = np.sort(results_5d)[::-1]  # 从大到小排序
trimmed_5d = sorted_5d[3:]  # 去掉最高的3个值

print("\n===== 剔除最高3个极端值后（5天）=====")
print(f"样本数量: {len(trimmed_5d)}")
print(f"平均涨跌幅: {trimmed_5d.mean():.2f}%")
print(f"上涨概率: {(trimmed_5d > 0).mean() * 100:.1f}%")

# 顺便打印一下被剔除的3个极端值具体是多少，方便你直观感受影响有多大
print(f"被剔除的3个极端值: {sorted_5d[:3]}")

print("\n===== 10天后表现 =====")
print("\n===== 10天后表现 =====")
print(f"样本数量: {len(results_10d)}")
print(f"平均涨跌幅: {results_10d.mean():.2f}%")
print(f"上涨概率: {(results_10d > 0).mean() * 100:.1f}%")
print(f"最大涨幅: {results_10d.max():.2f}%")
print(f"最大跌幅: {results_10d.min():.2f}%")

# ===== 第6步：对比一下"随机买入"的基准表现，作为参照系 =====
# 这一步很重要：如果我们的信号跑不赢"随便找一天买入"，说明这个指标没有实际价值
np.random.seed(42)
random_idx = np.random.choice(range(len(df) - 10), size=len(signal_dates), replace=False)

random_results_5d = []
for idx in random_idx:
    entry_price = df.loc[idx, "收盘"]
    if idx + 5 < len(df):
        price_5d_later = df.loc[idx + 5, "收盘"]
        random_results_5d.append((price_5d_later - entry_price) / entry_price * 100)

random_results_5d = np.array(random_results_5d)
print("\n===== 对照组：随机买入 5天后表现 =====")
print(f"平均涨跌幅: {random_results_5d.mean():.2f}%")
print(f"上涨概率: {(random_results_5d > 0).mean() * 100:.1f}%")