import akshare as ak
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
import time

window_length = 11
polyorder = 3
STOP_LOSS_PERCENT = 8.0


def get_stock_data_with_retry(stock_code, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 3 * (attempt + 1)
                print(f"请求失败，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"多次重试后仍然失败: {e}")
                return None


def judge_signal(prev_first, latest_first, latest_second):
    if prev_first < 0 and latest_first > 0:
        return "【潜在买入信号】一阶导数由负转正，短期趋势可能由跌转涨"
    elif latest_first > 0 and latest_second < 0:
        return "【注意】价格仍在上涨，但二阶导数转负，上涨动能可能正在减弱"
    elif latest_first < 0:
        return "【无买入信号】当前一阶导数为负，价格仍处于下跌趋势"
    else:
        return "【观察中】暂无明显的拐点信号"


def check_volume_confirmation(df):
    """
    检查成交量是否配合价格走势
    逻辑：比较最近5天平均成交量 vs 之前20天平均成交量
    """
    recent_volume = df["成交量"].tail(5).mean()
    baseline_volume = df["成交量"].tail(25).head(20).mean()

    volume_ratio = recent_volume / baseline_volume

    recent_price_change = df["收盘"].iloc[-1] - df["收盘"].iloc[-6]

    if recent_price_change > 0:
        if volume_ratio > 1.2:
            return "【成交量配合】价格上涨且成交量放大（近5日量能是前期的{:.1f}倍），资金推动力度较强，信号可信度较高".format(volume_ratio)
        elif volume_ratio < 0.8:
            return "【成交量背离】价格上涨但成交量萎缩（近5日量能只有前期的{:.1f}倍），上涨缺乏资金支撑，需警惕假突破".format(volume_ratio)
        else:
            return "【成交量中性】价格上涨，成交量变化不明显（{:.1f}倍），暂无特别提示".format(volume_ratio)
    else:
        if volume_ratio > 1.2:
            return "【放量下跌】价格下跌且成交量放大（近5日量能是前期的{:.1f}倍），抛压较重，趋势下行信号较强".format(volume_ratio)
        elif volume_ratio < 0.8:
            return "【缩量下跌】价格下跌但成交量萎缩（{:.1f}倍），抛压趋缓，可能存在止跌迹象".format(volume_ratio)
        else:
            return "【成交量中性】价格下跌，成交量变化不明显（{:.1f}倍），暂无特别提示".format(volume_ratio)


def check_stop_loss(entry_price, current_price, latest_second_deriv):
    loss_percent = (entry_price - current_price) / entry_price * 100
    messages = []

    if loss_percent >= STOP_LOSS_PERCENT:
        messages.append(f"【触发止损】当前亏损 {loss_percent:.2f}%，已达到设定止损线 {STOP_LOSS_PERCENT}%，建议考虑止损离场")
    elif loss_percent > 0:
        messages.append(f"当前浮亏 {loss_percent:.2f}%，尚未达到止损线（{STOP_LOSS_PERCENT}%），继续观察")
    else:
        messages.append(f"当前浮盈 {-loss_percent:.2f}%，暂无亏损")

    if latest_second_deriv < 0:
        messages.append("【预警】二阶导数为负，上涨/反弹动能正在减弱，即使未跌破止损线，也建议提高警惕")

    return messages


def analyze_stock(stock_code, entry_price=None):
    print(f"\n正在获取股票代码 {stock_code} 的最新数据...\n")

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")

    df = get_stock_data_with_retry(stock_code, start_date, end_date)

    if df is None or len(df) < 30:
        print("获取数据失败，或者数据量不足，无法分析。请检查股票代码是否正确。")
        return

    df = df.sort_values("日期").reset_index(drop=True)
    close_prices = df["收盘"].values

    df["灵敏_一阶导"] = savgol_filter(close_prices, window_length, polyorder, deriv=1)
    df["灵敏_二阶导"] = savgol_filter(close_prices, window_length, polyorder, deriv=2)

    df["稳健_平滑价"] = df["收盘"].ewm(span=10, adjust=False).mean()
    df["稳健_一阶导"] = df["稳健_平滑价"].diff()
    df["稳健_二阶导"] = df["稳健_一阶导"].diff()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    print("=" * 60)
    print(f"股票代码: {stock_code}")
    print(f"最新交易日: {latest['日期']}")
    print(f"最新收盘价: {latest['收盘']}")
    print("=" * 60)

    print("\n【方法A：灵敏版 Savitzky-Golay】")
    print(f"一阶导数: {latest['灵敏_一阶导']:.2f}")
    print(f"二阶导数: {latest['灵敏_二阶导']:.2f}")
    signal_a = judge_signal(prev["灵敏_一阶导"], latest["灵敏_一阶导"], latest["灵敏_二阶导"])
    print(f"信号判断: {signal_a}")

    print("\n【方法B：稳健版 EMA】")
    print(f"一阶导数: {latest['稳健_一阶导']:.2f}")
    print(f"二阶导数: {latest['稳健_二阶导']:.2f}")
    signal_b = judge_signal(prev["稳健_一阶导"], latest["稳健_一阶导"], latest["稳健_二阶导"])
    print(f"信号判断: {signal_b}")

    print("\n----- 综合参考建议 -----")
    if signal_a == signal_b:
        print("两种方法结论一致，信号可信度相对更高")
    else:
        print("两种方法结论不一致，说明当前正处于趋势转折的模糊地带，建议谨慎观望")

    # ===== 新增：成交量验证 =====
    print("\n----- 成交量验证 -----")
    volume_message = check_volume_confirmation(df)
    print(volume_message)

    if entry_price is not None:
        print("\n" + "=" * 60)
        print(f"----- 止损检查（假设买入价: {entry_price}）-----")
        print("=" * 60)
        stop_loss_messages = check_stop_loss(entry_price, latest["收盘"], latest["稳健_二阶导"])
        for msg in stop_loss_messages:
            print(msg)


if __name__ == "__main__":
    stock_code = input("请输入股票代码（例如 600519）: ").strip()

    entry_price_input = input("如果你已经持有这只股票，请输入你的买入价格（没有持有直接按回车跳过）: ").strip()

    entry_price = None
    if entry_price_input:
        try:
            entry_price = float(entry_price_input)
        except ValueError:
            print("输入的买入价格格式不正确，将跳过止损检查")

    analyze_stock(stock_code, entry_price)