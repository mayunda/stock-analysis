import streamlit as st
import akshare as ak
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
                time.sleep(3 * (attempt + 1))
            else:
                return None


def judge_signal(prev_first, latest_first, latest_second):
    if prev_first < 0 and latest_first > 0:
        return "潜在买入信号：一阶导数由负转正，短期趋势可能由跌转涨", "success"
    elif latest_first > 0 and latest_second < 0:
        return "注意：价格仍在上涨，但二阶导数转负，上涨动能可能正在减弱", "warning"
    elif latest_first < 0:
        return "无买入信号：当前一阶导数为负，价格仍处于下跌趋势", "error"
    else:
        return "观察中：暂无明显的拐点信号", "info"


def check_volume_confirmation(df):
    recent_volume = df["成交量"].tail(5).mean()
    baseline_volume = df["成交量"].tail(25).head(20).mean()
    volume_ratio = recent_volume / baseline_volume
    recent_price_change = df["收盘"].iloc[-1] - df["收盘"].iloc[-6]

    if recent_price_change > 0:
        if volume_ratio > 1.2:
            return f"成交量配合：价格上涨且成交量放大（近5日量能是前期的{volume_ratio:.1f}倍），资金推动力度较强", "success"
        elif volume_ratio < 0.8:
            return f"成交量背离：价格上涨但成交量萎缩（{volume_ratio:.1f}倍），需警惕假突破", "warning"
        else:
            return f"成交量中性：价格上涨，量能变化不明显（{volume_ratio:.1f}倍）", "info"
    else:
        if volume_ratio > 1.2:
            return f"放量下跌：价格下跌且成交量放大（{volume_ratio:.1f}倍），抛压较重", "error"
        elif volume_ratio < 0.8:
            return f"缩量下跌：价格下跌但成交量萎缩（{volume_ratio:.1f}倍），可能存在止跌迹象", "warning"
        else:
            return f"成交量中性：价格下跌，量能变化不明显（{volume_ratio:.1f}倍）", "info"


def check_stop_loss(entry_price, current_price, latest_second_deriv):
    loss_percent = (entry_price - current_price) / entry_price * 100
    messages = []

    if loss_percent >= STOP_LOSS_PERCENT:
        messages.append((f"触发止损：当前亏损 {loss_percent:.2f}%，已达到设定止损线 {STOP_LOSS_PERCENT}%", "error"))
    elif loss_percent > 0:
        messages.append((f"当前浮亏 {loss_percent:.2f}%，尚未达到止损线（{STOP_LOSS_PERCENT}%）", "warning"))
    else:
        messages.append((f"当前浮盈 {-loss_percent:.2f}%，暂无亏损", "success"))

    if latest_second_deriv < 0:
        messages.append(("预警：二阶导数为负，上涨/反弹动能正在减弱，建议提高警惕", "warning"))

    return messages


def show_message(text, level):
    if level == "success":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    elif level == "error":
        st.error(text)
    else:
        st.info(text)


# ========== 网页界面开始 ==========

st.set_page_config(page_title="A股导数信号分析", layout="wide")

st.title("A股K线导数分析工具")
st.caption("基于一阶导（涨跌速度）、二阶导（涨跌加速度）与成交量的技术面参考工具，仅供学习研究使用，不构成投资建议")
st.warning("⚠️ 免责声明：本工具基于历史价格数据的技术指标计算，仅用于技术学习交流，不构成任何投资建议。股市有风险，入市需谨慎，一切投资决策及后果由使用者自行承担。开发者不对因使用本工具产生的任何损失负责。")

col1, col2 = st.columns(2)
with col1:
    stock_code = st.text_input("请输入股票代码", value="600519", max_chars=6)
with col2:
    entry_price_input = st.text_input("买入价格（可选，不填则跳过止损检查）", value="")

analyze_button = st.button("开始分析", type="primary")

if analyze_button:
    if not stock_code or len(stock_code) != 6:
        st.error("请输入正确的6位股票代码")
    else:
        with st.spinner("正在获取数据，请稍候..."):
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            df = get_stock_data_with_retry(stock_code, start_date, end_date)

        if df is None or len(df) < 30:
            st.error("获取数据失败，或数据量不足，请检查股票代码是否正确，或稍后重试")
        else:
            df = df.sort_values("日期").reset_index(drop=True)
            close_prices = df["收盘"].values

            df["灵敏_一阶导"] = savgol_filter(close_prices, window_length, polyorder, deriv=1)
            df["灵敏_二阶导"] = savgol_filter(close_prices, window_length, polyorder, deriv=2)
            df["灵敏_平滑价"] = savgol_filter(close_prices, window_length, polyorder)

            df["稳健_平滑价"] = df["收盘"].ewm(span=10, adjust=False).mean()
            df["稳健_一阶导"] = df["稳健_平滑价"].diff()
            df["稳健_二阶导"] = df["稳健_一阶导"].diff()

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            st.subheader(f"{stock_code}　最新交易日: {latest['日期']}　最新收盘价: {latest['收盘']}")

            # ----- 图表 -----
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            dates = df["日期"].tail(60)

            axes[0].plot(dates, df["收盘"].tail(60), label="原始收盘价", alpha=0.4, color="gray")
            axes[0].plot(dates, df["稳健_平滑价"].tail(60), label="稳健平滑价(EMA)", color="blue")
            axes[0].set_title("价格走势（最近60个交易日）")
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(dates, df["稳健_一阶导"].tail(60), color="green")
            axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes[1].set_title("一阶导数（稳健版EMA）")
            axes[1].grid(True)

            axes[2].plot(dates, df["稳健_二阶导"].tail(60), color="red")
            axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes[2].set_title("二阶导数（稳健版EMA）")
            axes[2].grid(True)

            plt.xticks(rotation=45)
            step = max(len(dates) // 10, 1)
            axes[2].set_xticks(dates[::step])
            plt.tight_layout()
            st.pyplot(fig)

            # ----- 信号判断 -----
            st.markdown("### 信号判断")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**方法A：灵敏版 Savitzky-Golay**")
                st.write(f"一阶导数: {latest['灵敏_一阶导']:.2f}　二阶导数: {latest['灵敏_二阶导']:.2f}")
                msg_a, level_a = judge_signal(prev["灵敏_一阶导"], latest["灵敏_一阶导"], latest["灵敏_二阶导"])
                show_message(msg_a, level_a)

            with c2:
                st.markdown("**方法B：稳健版 EMA**")
                st.write(f"一阶导数: {latest['稳健_一阶导']:.2f}　二阶导数: {latest['稳健_二阶导']:.2f}")
                msg_b, level_b = judge_signal(prev["稳健_一阶导"], latest["稳健_一阶导"], latest["稳健_二阶导"])
                show_message(msg_b, level_b)

            if msg_a == msg_b:
                st.success("两种方法结论一致，信号可信度相对更高")
            else:
                st.warning("两种方法结论不一致，当前处于趋势转折的模糊地带，建议谨慎观望")

            st.markdown("### 成交量验证")
            vol_msg, vol_level = check_volume_confirmation(df)
            show_message(vol_msg, vol_level)

            if entry_price_input:
                try:
                    entry_price = float(entry_price_input)
                    st.markdown(f"### 止损检查（假设买入价: {entry_price}）")
                    stop_messages = check_stop_loss(entry_price, latest["收盘"], latest["稳健_二阶导"])
                    for msg, level in stop_messages:
                        show_message(msg, level)
                except ValueError:
                    st.error("买入价格格式不正确，已跳过止损检查")

            with st.expander("查看最近10天详细数据"):
                display_cols = ["日期", "收盘", "成交量", "灵敏_一阶导", "灵敏_二阶导", "稳健_一阶导", "稳健_二阶导"]
                st.dataframe(df[display_cols].tail(10), use_container_width=True)