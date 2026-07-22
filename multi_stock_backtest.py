import akshare as ak
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import time

# ===== 第1步：定义股票池 =====
stock_list = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "601318": "中国平安",
    "600036": "招商银行",
    "000333": "美的集团",
    "002415": "海康威视",
    "300750": "宁德时代",
    "601888": "中国中免",
    "600276": "恒瑞医药",
    "000651": "格力电器",
}

window_length = 11
polyorder = 3

# ===== 第2步：把"单只股票回测"包装成一个函数，方便重复调用 =====
def backtest_single_stock(stock_code, stock_name):
    """
    对单只股票进行信号检测和回测
    返回：这只股票所有信号的5天/10天涨跌幅列表
    """
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date="20220101",
            end_date="20241231",
            adjust="qfq"
        )
        df = df.sort_values("日期").reset_index(drop=True)

        # 数据太少（比如次新股）没法算，直接跳过
        if len(df) < 50:
            print(f"  [跳过] {stock_name}({stock_code}) 数据量过少")
            return [], []

        close_prices = df["收盘"].values
        df["一阶导"] = savgol_filter(close_prices, window_length, polyorder, deriv=1)
        df["一阶导_昨天"] = df["一阶导"].shift(1)
        df["买入信号"] = (df["一阶导_昨天"] < 0) & (df["一阶导"] > 0)

        signal_dates = df[df["买入信号"]].index.tolist()

        stock_results_5d = []
        stock_results_10d = []

        for idx in signal_dates:
            entry_price = df.loc[idx, "收盘"]
            if idx + 5 < len(df):
                price_5d = df.loc[idx + 5, "收盘"]
                stock_results_5d.append((price_5d - entry_price) / entry_price * 100)
            if idx + 10 < len(df):
                price_10d = df.loc[idx + 10, "收盘"]
                stock_results_10d.append((price_10d - entry_price) / entry_price * 100)

        print(f"  [完成] {stock_name}({stock_code}): 找到 {len(signal_dates)} 个信号")
        return stock_results_5d, stock_results_10d

    except Exception as e:
        print(f"  [错误] {stock_name}({stock_code}) 获取数据失败: {e}")
        return [], []


# ===== 第3步：循环跑所有股票，汇总结果 =====
all_results_5d = []
all_results_10d = []

print("开始批量回测...\n")

for code, name in stock_list.items():
    r5, r10 = backtest_single_stock(code, name)
    all_results_5d.extend(r5)
    all_results_10d.extend(r10)
    time.sleep(1)  # 每次请求间隔1秒，避免请求过于频繁被服务器限制

# ===== 第4步：汇总统计（所有股票合并在一起看整体表现） =====
all_results_5d = np.array(all_results_5d)
all_results_10d = np.array(all_results_10d)

print("\n" + "=" * 40)
print("===== 汇总结果：10只股票合计 =====")
print("=" * 40)

print(f"\n【5天后表现】")
print(f"总信号数量: {len(all_results_5d)}")
print(f"平均涨跌幅: {all_results_5d.mean():.2f}%")
print(f"上涨概率: {(all_results_5d > 0).mean() * 100:.1f}%")
print(f"收益标准差: {all_results_5d.std():.2f}%")

print(f"\n【10天后表现】")
print(f"总信号数量: {len(all_results_10d)}")
print(f"平均涨跌幅: {all_results_10d.mean():.2f}%")
print(f"上涨概率: {(all_results_10d > 0).mean() * 100:.1f}%")
print(f"收益标准差: {all_results_10d.std():.2f}%")

# ===== 第5步：整体的随机对照组（同样规模的随机样本） =====
# 这里简化处理：仍用茅台数据做对照基准，规模对齐到总信号数
df_base = ak.stock_zh_a_hist(
    symbol="600519", period="daily",
    start_date="20220101", end_date="20241231", adjust="qfq"
)
df_base = df_base.sort_values("日期").reset_index(drop=True)

np.random.seed(42)
n_samples = min(len(all_results_5d), len(df_base) - 10)
random_idx = np.random.choice(range(len(df_base) - 10), size=n_samples, replace=False)

random_results = []
for idx in random_idx:
    entry_price = df_base.loc[idx, "收盘"]
    price_5d = df_base.loc[idx + 5, "收盘"]
    random_results.append((price_5d - entry_price) / entry_price * 100)

random_results = np.array(random_results)
print(f"\n【对照组：随机买入（同等样本量）】")
print(f"平均涨跌幅: {random_results.mean():.2f}%")
print(f"上涨概率: {(random_results > 0).mean() * 100:.1f}%")