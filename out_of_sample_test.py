import akshare as ak
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import time

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


def get_signals_and_returns(stock_code, stock_name, start_date, end_date, max_retries=3):
    df = None
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 3 * (attempt + 1)
                print(f"  [重试 {attempt + 1}/{max_retries}] {stock_name}({stock_code}) 失败，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"  [彻底失败] {stock_name}({stock_code}): {e}")
                return []

    if df is None or len(df) < 50:
        return []

    df = df.sort_values("日期").reset_index(drop=True)
    close_prices = df["收盘"].values
    df["一阶导"] = savgol_filter(close_prices, window_length, polyorder, deriv=1)
    df["一阶导_昨天"] = df["一阶导"].shift(1)
    df["买入信号"] = (df["一阶导_昨天"] < 0) & (df["一阶导"] > 0)

    signal_dates = df[df["买入信号"]].index.tolist()

    results = []
    for idx in signal_dates:
        if idx + 5 < len(df):
            entry_price = df.loc[idx, "收盘"]
            price_5d = df.loc[idx + 5, "收盘"]
            results.append((price_5d - entry_price) / entry_price * 100)

    return results


def run_period_test(period_name, start_date, end_date):
    print(f"\n{'=' * 50}")
    print(f"正在测试：{period_name} ({start_date} ~ {end_date})")
    print(f"{'=' * 50}")

    all_results = []
    for code, name in stock_list.items():
        results = get_signals_and_returns(code, name, start_date, end_date)
        print(f"  {name}({code}): {len(results)} 个信号")
        all_results.extend(results)
        time.sleep(2)

    all_results = np.array(all_results)

    print(f"\n【{period_name} 汇总结果】")
    print(f"总信号数量: {len(all_results)}")
    if len(all_results) > 0:
        print(f"平均涨跌幅: {all_results.mean():.2f}%")
        print(f"上涨概率: {(all_results > 0).mean() * 100:.1f}%")
        print(f"收益标准差: {all_results.std():.2f}%")

    return all_results


train_results = run_period_test("训练期（2022-2023）", "20220101", "20231231")
test_results = run_period_test("样本外验证期（2024）", "20240101", "20241231")

print(f"\n{'=' * 50}")
print("最终对比总结")
print(f"{'=' * 50}")
print(f"{'指标':<15}{'训练期(22-23)':<20}{'验证期(2024)':<20}")
print(f"{'平均涨跌幅':<15}{train_results.mean():<20.2f}{test_results.mean():<20.2f}")
print(f"{'上涨概率(%)':<15}{(train_results > 0).mean()*100:<20.1f}{(test_results > 0).mean()*100:<20.1f}")
print(f"{'信号数量':<15}{len(train_results):<20}{len(test_results):<20}")