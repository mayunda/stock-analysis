import streamlit as st
import akshare as ak
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
import time
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import mplfinance as mpf

font_path = "fonts/SimHei.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

window_length = 11
polyorder = 3

ATR_PERIOD = 14
ATR_MULTIPLIER = 2.5

USERS_FILE = "users.json"


# ========== 账户系统 ==========

def load_users():
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)


def check_login(username, password):
    users = load_users()
    if username in users and users[username]["password"] == password:
        return users[username]
    return None


def login_page():
    st.title("登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    login_button = st.button("登录", type="primary")

    if login_button:
        user_info = check_login(username, password)
        if user_info:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = user_info["role"]
            st.session_state["display_name"] = user_info["display_name"]
            st.rerun()
        else:
            st.error("用户名或密码错误")


# ========== 数据获取（含主备数据源切换）==========

def get_stock_code_with_prefix(stock_code):
    if stock_code.startswith("6"):
        return "sh" + stock_code
    else:
        return "sz" + stock_code


def get_data_from_eastmoney(stock_code, start_date, end_date):
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )
    return df


def get_data_from_tencent(stock_code, start_date, end_date):
    prefixed_code = get_stock_code_with_prefix(stock_code)
    df = ak.stock_zh_a_hist_tx(symbol=prefixed_code)

    df = df.rename(columns={
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "amount": "成交量"
    })

    df["日期"] = pd.to_datetime(df["日期"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df["日期"] >= start) & (df["日期"] <= end)]
    df["日期"] = df["日期"].dt.strftime("%Y-%m-%d")

    return df


def get_stock_data_with_retry(stock_code, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = get_data_from_eastmoney(stock_code, start_date, end_date)
            return df, "东方财富"
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))

    for attempt in range(max_retries):
        try:
            df = get_data_from_tencent(stock_code, start_date, end_date)
            return df, "腾讯财经（备用源）"
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))

    return None, None


def get_stock_name(stock_code):
    try:
        name_df = ak.stock_info_a_code_name()
        result = name_df[name_df["code"] == stock_code]
        if not result.empty:
            return result["name"].values[0]
        return None
    except Exception:
        return None


# ========== 资金面数据：股东增减持 + 北向资金 ==========

def get_shareholder_signal(stock_code):
    """
    查询最近90天内的股东增减持公告，判断是增持还是减持
    返回：(信号: "增持为主"/"减持为主"/"增减持相当"/None, 详细说明文字)
    """
    try:
        df = ak.stock_ggcg_em(symbol="全部")
        stock_records = df[df["代码"] == stock_code].copy()

        if stock_records.empty:
            return None, "近期无股东增减持公告记录"

        stock_records["公告日"] = pd.to_datetime(stock_records["公告日"])
        stock_records = stock_records.sort_values("公告日", ascending=False)

        cutoff_date = datetime.now() - timedelta(days=90)
        recent_records = stock_records[stock_records["公告日"] >= cutoff_date]

        if recent_records.empty:
            latest_record = stock_records.iloc[0]
            return None, f"最近一条公告是 {latest_record['公告日'].strftime('%Y-%m-%d')}（超过90天，视为无近期信号）"

        increase_count = (recent_records["持股变动信息-增减"] == "增持").sum()
        decrease_count = (recent_records["持股变动信息-增减"] == "减持").sum()
        detail = f"近90天内共{len(recent_records)}条公告（增持{increase_count}次，减持{decrease_count}次）"

        if increase_count > decrease_count:
            return "增持为主", detail
        elif decrease_count > increase_count:
            return "减持为主", detail
        else:
            return "增减持相当", detail

    except Exception as e:
        return None, f"查询失败: {e}"


def get_northbound_signal(stock_code):
    """
    查询最近5天的北向资金持股变化，判断资金是净流入还是净流出
    返回：(信号: "净流入"/"净流出"/"基本持平"/None, 详细说明文字)
    """
    try:
        df = ak.stock_hsgt_individual_em(symbol=stock_code)
        if df is None or len(df) < 5:
            return None, "北向资金数据不足"

        recent = df.tail(5)
        net_change = recent["今日增持资金"].sum()
        detail = f"最近5个交易日北向资金净变化约 {net_change / 1e8:.2f} 亿元"

        if net_change > 0:
            return "净流入", detail
        elif net_change < 0:
            return "净流出", detail
        else:
            return "基本持平", detail

    except Exception as e:
        return None, f"查询失败: {e}"


def get_capital_conclusion(stock_code):
    """
    综合股东增减持 + 北向资金，给出资金面结论：积极 / 中性 / 消极
    返回：(结论, level, 详细依据列表)
    """
    shareholder_signal, shareholder_detail = get_shareholder_signal(stock_code)
    northbound_signal, northbound_detail = get_northbound_signal(stock_code)

    score = 0
    details = []

    if shareholder_signal == "增持为主":
        score += 1
        details.append(f"股东公告：{shareholder_detail}（偏积极）")
    elif shareholder_signal == "减持为主":
        score -= 1
        details.append(f"股东公告：{shareholder_detail}（偏消极）")
    else:
        details.append(f"股东公告：{shareholder_detail}")

    if northbound_signal == "净流入":
        score += 1
        details.append(f"北向资金：{northbound_detail}（偏积极）")
    elif northbound_signal == "净流出":
        score -= 1
        details.append(f"北向资金：{northbound_detail}（偏消极）")
    else:
        details.append(f"北向资金：{northbound_detail}")

    if score > 0:
        conclusion, level = "积极", "success"
    elif score < 0:
        conclusion, level = "消极", "error"
    else:
        conclusion, level = "中性", "info"

    return conclusion, level, details


# ========== 技术面信号判断 ==========

def judge_signal(prev_first, latest_first, latest_second):
    if prev_first < 0 and latest_first > 0:
        return "潜在买入信号：动量由负转正，短期趋势可能由跌转涨", "success"
    elif latest_first > 0 and latest_second < 0:
        return "注意：价格仍在上涨，但曲率转负，上涨动能可能正在减弱", "warning"
    elif latest_first < 0:
        return "无买入信号：当前动量为负，价格仍处于下跌趋势", "error"
    else:
        return "观察中：暂无明显的拐点信号", "info"


def get_technical_conclusion(prev_first, latest_first, latest_second, is_stop_loss_triggered=False):
    """
    技术面结论：买入 / 观望 / 卖出
    逻辑类似物理中的位移-速度-加速度关系：
    - 动量(一阶导)由负转正 = 波谷拐点，是买入时机；若曲率(二阶导)同时为正，说明确实在加速向上，信号更强
    - 动量由正转负 = 波峰拐点，是卖出/止盈时机
    - 已触发止损线，直接判定卖出
    """
    if is_stop_loss_triggered:
        return "卖出", "error", ["已触发止损线，建议考虑离场，控制风险"]

    reasons = []

    if prev_first < 0 and latest_first > 0:
        if latest_second > 0:
            reasons.append("动量由负转正，且曲率为正，符合“波谷”特征，是相对理想的买入时机")
            return "买入", "success", reasons
        else:
            reasons.append("动量刚由负转正，但曲率尚未转正，拐点信号还不够扎实，建议谨慎小仓位观察")
            return "观望", "warning", reasons

    if prev_first > 0 and latest_first < 0:
        reasons.append("动量由正转负，符合“波峰”特征，上涨动能已经耗尽，建议考虑卖出/止盈")
        return "卖出", "error", reasons

    if latest_first < 0:
        reasons.append("当前动量为负，价格仍处于下跌趋势中，暂无买入依据")
        return "观望", "error", reasons

    if latest_first > 0 and latest_second < 0:
        reasons.append("价格仍在上涨，但曲率已转负，上涨动能正在减弱，需警惕见顶风险")
        return "观望", "warning", reasons

    reasons.append("暂无明显拐点信号，趋势不够清晰")
    return "观望", "info", reasons


def get_overall_conclusion(technical_conclusion, capital_conclusion):
    """
    综合技术面结论（买入/观望/卖出）与资金面结论（积极/中性/消极），给出最终综合结论
    """
    if technical_conclusion == "买入" and capital_conclusion == "积极":
        return "买入", "success", "技术面与资金面均积极，信号一致性较强"
    elif technical_conclusion == "买入" and capital_conclusion == "消极":
        return "观望", "warning", "技术面显示买入信号，但资金面偏消极，存在矛盾，建议谨慎"
    elif technical_conclusion == "卖出" and capital_conclusion == "消极":
        return "卖出", "error", "技术面与资金面均偏消极，信号一致性较强"
    elif technical_conclusion == "卖出" and capital_conclusion == "积极":
        return "观望", "warning", "技术面显示卖出信号，但资金面偏积极，存在矛盾，建议谨慎"
    elif technical_conclusion == "买入" and capital_conclusion == "中性":
        return "买入", "success", "技术面积极，资金面中性，暂无明显矛盾"
    elif technical_conclusion == "卖出" and capital_conclusion == "中性":
        return "卖出", "error", "技术面消极，资金面中性，暂无明显矛盾"
    else:
        return "观望", "info", "技术面与资金面均无明确一致方向，建议继续观察"


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


def calculate_atr(df, period=ATR_PERIOD):
    high = df["最高"]
    low = df["最低"]
    prev_close = df["收盘"].shift(1)

    range1 = high - low
    range2 = (high - prev_close).abs()
    range3 = (low - prev_close).abs()

    true_range = pd.concat([range1, range2, range3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def check_stop_loss(entry_price, current_price, latest_second_deriv, atr_value):
    loss_percent = (entry_price - current_price) / entry_price * 100
    dynamic_stop_loss_percent = (atr_value * ATR_MULTIPLIER) / entry_price * 100

    messages = []
    messages.append((f"该股票近期波动率(ATR)对应的动态止损线约为: {dynamic_stop_loss_percent:.2f}%（ATR×{ATR_MULTIPLIER}）", "info"))

    is_triggered = loss_percent >= dynamic_stop_loss_percent

    if is_triggered:
        messages.append((f"触发止损：当前亏损 {loss_percent:.2f}%，已超过该股票的动态止损线 {dynamic_stop_loss_percent:.2f}%，建议考虑止损离场", "error"))
    elif loss_percent > 0:
        messages.append((f"当前浮亏 {loss_percent:.2f}%，尚未达到动态止损线（{dynamic_stop_loss_percent:.2f}%），继续观察", "warning"))
    else:
        messages.append((f"当前浮盈 {-loss_percent:.2f}%，暂无亏损", "success"))

    if latest_second_deriv < 0:
        messages.append(("预警：曲率为负，上涨/反弹动能正在减弱，建议提高警惕", "warning"))

    return messages, is_triggered


def show_message(text, level):
    if level == "success":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    elif level == "error":
        st.error(text)
    else:
        st.info(text)


def render_conclusion_box(title, conclusion, level, reason_text):
    color_map = {"success": "#d4edda", "warning": "#fff3cd", "error": "#f8d7da", "info": "#d1ecf1"}
    text_color_map = {"success": "#155724", "warning": "#856404", "error": "#721c24", "info": "#0c5460"}

    bg = color_map.get(level, "#d1ecf1")
    tc = text_color_map.get(level, "#0c5460")

    st.markdown(
        f"""
        <div style="background-color:{bg}; padding:16px; border-radius:10px; margin-bottom:10px;">
            <span style="font-size:15px; color:{tc};">{title}</span><br>
            <span style="font-size:22px; font-weight:bold; color:{tc};">{conclusion}</span>
            <br><br>
            <span style="color:{tc}; font-size:14px;">{reason_text}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ========== 网页界面开始 ==========

st.set_page_config(page_title="A股动量曲率分析", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()

st.sidebar.write(f"当前登录用户：{st.session_state['display_name']}（{st.session_state['role']}）")
if st.sidebar.button("退出登录"):
    st.session_state["logged_in"] = False
    st.rerun()

st.title("A股K线动量与曲率分析工具")
st.caption("基于动量（价格变化速度）、曲率（价格变化的弯曲程度）、成交量与资金面（股东增减持、北向资金）的技术面参考工具，仅供学习研究使用，不构成投资建议")
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
            stock_name = get_stock_name(stock_code)
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            df, data_source = get_stock_data_with_retry(stock_code, start_date, end_date)

        if df is None or len(df) < 30:
            st.error("获取数据失败，或数据量不足，请检查股票代码是否正确，或稍后重试")
        else:
            if data_source == "腾讯财经（备用源）":
                st.info(f"提示：主数据源暂时不可用，已自动切换到 {data_source} 获取数据")

            df = df.sort_values("日期").reset_index(drop=True)
            close_prices = df["收盘"].values

            df["灵敏_一阶导"] = savgol_filter(close_prices, window_length, polyorder, deriv=1)
            df["灵敏_二阶导"] = savgol_filter(close_prices, window_length, polyorder, deriv=2)
            df["灵敏_平滑价"] = savgol_filter(close_prices, window_length, polyorder)

            df["稳健_平滑价"] = df["收盘"].ewm(span=10, adjust=False).mean()
            df["稳健_一阶导"] = df["稳健_平滑价"].diff()
            df["稳健_二阶导"] = df["稳健_一阶导"].diff()

            df["ATR"] = calculate_atr(df)

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            display_name = f"{stock_name}（{stock_code}）" if stock_name else stock_code
            st.subheader(f"{display_name}　最新交易日: {latest['日期']}　最新收盘价: {latest['收盘']}")

            # ===== 先判断是否已触发止损（若填了买入价）=====
            is_stop_loss_triggered = False
            latest_atr = df["ATR"].iloc[-1]
            entry_price = None
            if entry_price_input:
                try:
                    entry_price = float(entry_price_input)
                    _, is_stop_loss_triggered = check_stop_loss(entry_price, latest["收盘"], latest["稳健_二阶导"], latest_atr)
                except ValueError:
                    entry_price = None

            # ===== 结论1：技术面结论（自动显示，速度快）=====
            technical_conclusion, technical_level, technical_reasons = get_technical_conclusion(
                prev["稳健_一阶导"], latest["稳健_一阶导"], latest["稳健_二阶导"], is_stop_loss_triggered
            )

            st.markdown("### 技术面结论（动量+曲率）")
            render_conclusion_box("技术面结论", technical_conclusion, technical_level, "；".join(technical_reasons))

            # ===== 资金面结论：改为手动触发，避免拖慢默认分析速度 =====
            st.markdown("### 资金面结论（可选，需单独查询）")
            st.caption("股东增减持公告与北向资金查询耗时较长，默认不自动执行，点击下方按钮才会查询")

            capital_button = st.button("查询资金面数据（股东增减持 + 北向资金）", key=f"capital_btn_{stock_code}")

            capital_cache_key = f"capital_result_{stock_code}"

            if capital_button:
                with st.spinner("正在查询股东增减持与北向资金数据，可能需要一些时间..."):
                    capital_conclusion, capital_level, capital_details = get_capital_conclusion(stock_code)
                    st.session_state[capital_cache_key] = (capital_conclusion, capital_level, capital_details)

            if capital_cache_key in st.session_state:
                capital_conclusion, capital_level, capital_details = st.session_state[capital_cache_key]

                overall_conclusion, overall_level, overall_reason = get_overall_conclusion(technical_conclusion, capital_conclusion)

                col_b, col_c = st.columns(2)
                with col_b:
                    render_conclusion_box("资金面结论（股东+北向资金）", capital_conclusion, capital_level, "；".join(capital_details))
                with col_c:
                    render_conclusion_box("综合结论", overall_conclusion, overall_level, overall_reason)

            st.caption("以上结论仅基于历史数据与公开信息的规则计算，不构成投资建议，最终决策请结合自身判断")

            # ===== 蜡烛图 =====
            try:
                plot_df = df.tail(60).copy()
                plot_df["日期"] = pd.to_datetime(plot_df["日期"])
                plot_df = plot_df.set_index("日期")
                plot_df = plot_df.rename(columns={
                    "开盘": "Open",
                    "最高": "High",
                    "最低": "Low",
                    "收盘": "Close",
                    "成交量": "Volume"
                })
                plot_df = plot_df[["Open", "High", "Low", "Close", "Volume", "稳健_平滑价"]].dropna()

                ema_line = mpf.make_addplot(
                    plot_df["稳健_平滑价"],
                    color="blue",
                    width=1.2
                )

                mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
                s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='yahoo', rc={'font.family': 'SimHei'})

                fig, axlist = mpf.plot(
                    plot_df,
                    type='candle',
                    addplot=ema_line,
                    style=s,
                    volume=True,
                    returnfig=True,
                    figsize=(12, 6),
                    title="K线走势（最近60个交易日，红涨绿跌）",
                    ylabel="价格",
                    ylabel_lower="成交量",
                    datetime_format="%Y-%m-%d",
                    xrotation=45
                )
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"蜡烛图绘制失败，改用折线图展示：{e}")
                fig_fallback, ax_fallback = plt.subplots(figsize=(12, 4))
                ax_fallback.plot(df["日期"].tail(60), df["收盘"].tail(60), label="收盘价", color="blue")
                ax_fallback.set_title("价格走势（最近60个交易日）")
                ax_fallback.legend()
                ax_fallback.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig_fallback)

            # ===== 动量、曲率单独用折线图展示 =====
            fig2, axes2 = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            dates = df["日期"].tail(60)

            axes2[0].plot(dates, df["稳健_一阶导"].tail(60), color="green")
            axes2[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes2[0].set_title("动量（稳健版EMA）")
            axes2[0].grid(True)

            axes2[1].plot(dates, df["稳健_二阶导"].tail(60), color="red")
            axes2[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes2[1].set_title("曲率（稳健版EMA）")
            axes2[1].grid(True)

            plt.xticks(rotation=45)
            step = max(len(dates) // 10, 1)
            axes2[1].set_xticks(dates[::step])
            plt.tight_layout()
            st.pyplot(fig2)

            st.markdown("### 信号判断（详细版）")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**方法A：灵敏版 Savitzky-Golay**")
                st.write(f"动量: {latest['灵敏_一阶导']:.2f}　曲率: {latest['灵敏_二阶导']:.2f}")
                msg_a, level_a = judge_signal(prev["灵敏_一阶导"], latest["灵敏_一阶导"], latest["灵敏_二阶导"])
                show_message(msg_a, level_a)

            with c2:
                st.markdown("**方法B：稳健版 EMA**")
                st.write(f"动量: {latest['稳健_一阶导']:.2f}　曲率: {latest['稳健_二阶导']:.2f}")
                msg_b, level_b = judge_signal(prev["稳健_一阶导"], latest["稳健_一阶导"], latest["稳健_二阶导"])
                show_message(msg_b, level_b)

            if msg_a == msg_b:
                st.success("两种方法结论一致，信号可信度相对更高")
            else:
                st.warning("两种方法结论不一致，当前处于趋势转折的模糊地带，建议谨慎观望")

            st.markdown("### 成交量验证")
            vol_msg, vol_level = check_volume_confirmation(df)
            show_message(vol_msg, vol_level)

            if entry_price is not None:
                st.markdown(f"### 止损检查（假设买入价: {entry_price}）")
                stop_messages, _ = check_stop_loss(entry_price, latest["收盘"], latest["稳健_二阶导"], latest_atr)
                for msg, level in stop_messages:
                    show_message(msg, level)
            elif entry_price_input:
                st.error("买入价格格式不正确，已跳过止损检查")

            with st.expander("查看最近10天详细数据"):
                display_cols = ["日期", "收盘", "成交量", "灵敏_一阶导", "灵敏_二阶导", "稳健_一阶导", "稳健_二阶导", "ATR"]
                display_df = df[display_cols].tail(10).rename(columns={
                    "灵敏_一阶导": "灵敏_动量",
                    "灵敏_二阶导": "灵敏_曲率",
                    "稳健_一阶导": "稳健_动量",
                    "稳健_二阶导": "稳健_曲率",
                })
                st.dataframe(display_df, use_container_width=True)


# ========== 批量监控 ==========

st.markdown("---")
st.markdown("## 批量监控")
st.caption("一次输入多个股票代码（用英文逗号分隔），快速查看每只股票的技术面（动量+曲率）信号摘要。资金面数据查询较慢，暂不包含在批量模式中，如需查看请到上方单只股票分析中单独查询")

batch_input = st.text_area(
    "请输入多个股票代码，用英文逗号分隔，例如：600519,000858,601318",
    value="",
    height=80
)

batch_button = st.button("批量分析", type="secondary")

if batch_button:
    if not batch_input.strip():
        st.error("请至少输入一个股票代码")
    else:
        codes = [c.strip() for c in batch_input.split(",") if c.strip()]

        if len(codes) > 15:
            st.warning(f"一次最多建议分析15只股票，你输入了{len(codes)}只，可能耗时较长")

        results_table = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, code in enumerate(codes):
            status_text.text(f"正在分析 {code} ({i+1}/{len(codes)}) ...")

            if len(code) != 6:
                results_table.append({
                    "代码": code, "名称": "-", "最新价": "-",
                    "技术面结论": "代码格式错误", "曲率": "-", "成交量": "-"
                })
                progress_bar.progress((i + 1) / len(codes))
                continue

            b_name = get_stock_name(code)
            b_end_date = datetime.now().strftime("%Y%m%d")
            b_start_date = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            b_df, b_source = get_stock_data_with_retry(code, b_start_date, b_end_date)

            if b_df is None or len(b_df) < 30:
                results_table.append({
                    "代码": code, "名称": b_name if b_name else "-", "最新价": "-",
                    "技术面结论": "数据获取失败", "曲率": "-", "成交量": "-"
                })
                progress_bar.progress((i + 1) / len(codes))
                continue

            b_df = b_df.sort_values("日期").reset_index(drop=True)

            b_df["稳健_平滑价"] = b_df["收盘"].ewm(span=10, adjust=False).mean()
            b_df["稳健_一阶导"] = b_df["稳健_平滑价"].diff()
            b_df["稳健_二阶导"] = b_df["稳健_一阶导"].diff()

            b_latest = b_df.iloc[-1]
            b_prev = b_df.iloc[-2]

            b_technical, b_tech_level, _ = get_technical_conclusion(
                b_prev["稳健_一阶导"], b_latest["稳健_一阶导"], b_latest["稳健_二阶导"]
            )

            b_vol_msg, b_vol_level = check_volume_confirmation(b_df)
            vol_short = "放量" if "放量" in b_vol_msg else ("缩量" if ("缩量" in b_vol_msg or "萎缩" in b_vol_msg) else "平稳")

            icon_map = {"买入": "🟢 买入", "观望": "🟡 观望", "卖出": "🔴 卖出",
                        "积极": "🟢 积极", "中性": "⚪ 中性", "消极": "🔴 消极"}

            results_table.append({
                "代码": code,
                "名称": b_name if b_name else "-",
                "最新价": round(b_latest["收盘"], 2),
                "技术面结论": icon_map.get(b_technical, b_technical),
                "曲率": round(b_latest["稳健_二阶导"], 2),
                "成交量": vol_short
            })

            progress_bar.progress((i + 1) / len(codes))
            time.sleep(1)

        status_text.text("分析完成")
        result_df = pd.DataFrame(results_table)
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.caption("提示：批量模式仅展示核心摘要，如需查看某只股票的详细图表和止损分析，请在上方单独输入该股票代码分析")