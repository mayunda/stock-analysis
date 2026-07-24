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
    """
    返回值三种情况：
    - (user_info, "ok")：账号密码正确且账户已启用
    - (None, "disabled")：账号密码正确，但账户已被禁用
    - (None, "invalid")：用户名或密码错误
    """
    users = load_users()
    if username in users and users[username]["password"] == password:
        # 兼容旧数据：如果账户信息里没有"启用"这个字段，默认视为已启用
        is_enabled = users[username].get("启用", True)
        if is_enabled:
            return users[username], "ok"
        else:
            return None, "disabled"
    return None, "invalid"


def login_page():
    st.title("登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    login_button = st.button("登录", type="primary")

    if login_button:
        user_info, status = check_login(username, password)
        if status == "ok":
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = user_info["role"]
            st.session_state["display_name"] = user_info["display_name"]
            st.rerun()
        elif status == "disabled":
            st.error("该账户已被管理员禁用，暂时无法登录")
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


def get_data_from_sina(stock_code, start_date, end_date):
    """
    第三备用数据源：新浪财经
    """
    prefixed_code = get_stock_code_with_prefix(stock_code)
    df = ak.stock_zh_a_daily(symbol=prefixed_code)

    df = df.reset_index()
    df = df.rename(columns={
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "volume": "成交量"
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

    for attempt in range(max_retries):
        try:
            df = get_data_from_sina(stock_code, start_date, end_date)
            return df, "新浪财经（备用源）"
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

def evaluate_version_conclusion(prev_first, latest_first, prev_second, latest_second):
    """
    对单一版本（灵敏版 或 稳健版）给出 买入/观望/卖出 结论。
    逻辑基于物理中位移-速度-加速度的关系，动量(一阶导)和曲率(二阶导)组合判断，而非只看动量单一指标：

    - 强买入（波谷已过）：动量由负转正 —— 已经越过波谷拐点，确认反转
    - 早期买入（接近波谷）：动量为负、曲率也为负，但两者都在向0靠拢（跌势正在减速，尚未反转但正在触底）
    - 强卖出（波峰已过）：动量由正转负 —— 已经越过波峰拐点，确认见顶
    - 早期卖出（接近波峰）：动量为正、曲率也为正，但两者都在向0靠拢（涨势正在减速，尚未转跌但动能耗尽）
    - 其他情况：观望
    """
    # 强买入：已越过波谷
    if prev_first < 0 and latest_first > 0:
        return "买入", "动量已由负转正，价格已越过波谷拐点，是确认性的买入信号"

    # 强卖出：已越过波峰
    if prev_first > 0 and latest_first < 0:
        return "卖出", "动量已由正转负，价格已越过波峰拐点，是确认性的卖出信号"

    # 早期买入：动量、曲率同为负值，且都在向0靠拢（跌势减速，接近波谷）
    if latest_first < 0 and latest_second < 0 and latest_second > prev_second and latest_first > prev_first:
        return "买入", "动量与曲率均为负值，但都在向0靠拢，跌势正在减速，接近波谷，属于提前信号"

    # 早期卖出：动量、曲率同为正值，且都在向0靠拢（涨势减速，接近波峰）
    if latest_first > 0 and latest_second > 0 and latest_second < prev_second and latest_first < prev_first:
        return "卖出", "动量与曲率均为正值，但都在向0靠拢，涨势正在减速，接近波峰，属于提前信号"

    return "观望", "动量与曲率未呈现明显的波峰/波谷趋势特征，暂不具备明确的买卖依据"


def combine_version_conclusions(sensitive_conclusion, robust_conclusion):
    """
    合并灵敏版与稳健版各自的结论，得到统一的技术面结论
    规则：只要有一个版本给出买入/卖出，且另一个版本不是相反方向，就采纳；
    若两个版本方向相反（一个买入一个卖出），判定为观望并提示矛盾
    """
    if sensitive_conclusion == robust_conclusion:
        return sensitive_conclusion, False  # 两版本一致，无矛盾

    if {"买入", "卖出"} == {sensitive_conclusion, robust_conclusion}:
        return "观望", True  # 两版本方向相反，矛盾

    # 一个是买入/卖出，另一个是观望 -> 采纳非观望的那个
    if sensitive_conclusion != "观望":
        return sensitive_conclusion, False
    if robust_conclusion != "观望":
        return robust_conclusion, False

    return "观望", False


def get_technical_conclusion(sensitive_prev_first, sensitive_latest_first, sensitive_prev_second, sensitive_latest_second,
                              robust_prev_first, robust_latest_first, robust_prev_second, robust_latest_second,
                              is_stop_loss_triggered=False):
    """
    技术面结论：买入 / 观望 / 卖出
    分别对灵敏版、稳健版计算各自结论，再合并成统一的技术面结论
    """
    if is_stop_loss_triggered:
        return "卖出", "error", ["已触发止损线，建议考虑离场，控制风险"], "-", "-"

    sensitive_conclusion, sensitive_reason = evaluate_version_conclusion(
        sensitive_prev_first, sensitive_latest_first, sensitive_prev_second, sensitive_latest_second
    )
    robust_conclusion, robust_reason = evaluate_version_conclusion(
        robust_prev_first, robust_latest_first, robust_prev_second, robust_latest_second
    )

    final_conclusion, is_conflict = combine_version_conclusions(sensitive_conclusion, robust_conclusion)

    level_map = {"买入": "success", "卖出": "error", "观望": "info"}
    level = level_map[final_conclusion]

    reasons = [f"灵敏版：{sensitive_reason}", f"稳健版：{robust_reason}"]
    if is_conflict:
        reasons.append("两个版本结论方向相反，存在分歧，建议谨慎观望")
        level = "warning"

    return final_conclusion, level, reasons, sensitive_conclusion, robust_conclusion


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


# ========== 智能推荐功能 ==========

RECOMMEND_CACHE_FILE = "recommend_cache.json"
RECOMMEND_SCAN_POOL_SIZE = 300
RECOMMEND_TARGET_COUNT = 5
RECOMMEND_VALID_HOURS = 3


def load_recommend_cache():
    try:
        with open(RECOMMEND_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_recommend_cache(data):
    try:
        with open(RECOMMEND_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception:
        pass


def is_cache_valid(cache):
    """
    缓存只有在“上次成功凑够目标数量”的前提下，才受3小时冷却限制。
    如果上次结果不完整（数量不足5只），视为无效缓存，允许随时重新扫描。
    """
    if cache is None or "timestamp" not in cache or "results" not in cache:
        return False

    if len(cache["results"]) < RECOMMEND_TARGET_COUNT:
        return False

    cache_time = datetime.fromisoformat(cache["timestamp"])
    elapsed_hours = (datetime.now() - cache_time).total_seconds() / 3600
    return elapsed_hours < RECOMMEND_VALID_HOURS


def get_candidate_pool(max_retries=3):
    """
    获取全市场股票快照，按价格升序排序（优先低价股，但绝不排除任何价格区间的股票）
    主数据源：东方财富；失败则切换到备用源：新浪财经
    返回排序后的完整DataFrame，包含代码、名称、最新价（代码统一为不带交易所前缀的纯数字格式）
    """
    spot_df = None
    source_used = None

    # ===== 主数据源：东方财富 =====
    for attempt in range(max_retries):
        try:
            spot_df = ak.stock_zh_a_spot_em()
            spot_df = spot_df[["代码", "名称", "最新价"]].dropna()
            source_used = "东方财富"
            break
        except Exception:
            spot_df = None
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))

    # ===== 备用数据源：新浪财经 =====
    if spot_df is None:
        for attempt in range(max_retries):
            try:
                sina_df = ak.stock_zh_a_spot()
                sina_df = sina_df[["代码", "名称", "最新价"]].dropna()
                # 新浪返回的代码带交易所前缀（sh/sz/bj），统一去除前缀，只保留纯数字部分
                sina_df["代码"] = sina_df["代码"].str.replace(r"^[a-zA-Z]+", "", regex=True)
                spot_df = sina_df
                source_used = "新浪财经（备用源）"
                break
            except Exception:
                spot_df = None
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))

    if spot_df is None:
        return pd.DataFrame(columns=["代码", "名称", "最新价"])

    spot_df = spot_df[spot_df["最新价"] > 0]
    # 按价格升序排序：低价股（尤其10元、20元以内）优先被扫描到，但不设置任何价格上限排除
    spot_df = spot_df.sort_values("最新价", ascending=True).reset_index(drop=True)
    return spot_df


def scan_for_recommendations(progress_callback=None):
    """
    扫描候选股票池（按价格从低到高排序），找出灵敏版或稳健版任一出现买入信号的股票。
    优先扫描前RECOMMEND_SCAN_POOL_SIZE只（价格最低的一批），如果不够目标数量，
    继续向后扩大扫描范围（不排除任何价格区间），直到凑够目标数量或达到硬性扫描上限。
    progress_callback: 可选的回调函数，用于更新进度提示，接收(当前扫描数, 预计总数, 已找到数)
    """
    all_candidates = get_candidate_pool()

    if all_candidates.empty:
        return []

    found = []
    max_scan_limit = len(all_candidates)  # 不设硬性上限，扫描全市场直到凑够目标数量为止

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")

    for i in range(max_scan_limit):
        row = all_candidates.iloc[i]
        code = row["代码"]
        name = row["名称"]

        # 进度显示：优先按预期的RECOMMEND_SCAN_POOL_SIZE估算，超出后按当前扩展范围显示
        estimated_total = max(RECOMMEND_SCAN_POOL_SIZE, i + 1)
        if progress_callback:
            progress_callback(i + 1, estimated_total, len(found))

        if len(found) >= RECOMMEND_TARGET_COUNT:
            break

        df, source = get_stock_data_with_retry(code, start_date, end_date, max_retries=1)
        if df is None or len(df) < 30:
            continue

        try:
            df = df.sort_values("日期").reset_index(drop=True)
            close_prices = df["收盘"].values

            sensitive_first = savgol_filter(close_prices, window_length, polyorder, deriv=1)
            sensitive_second = savgol_filter(close_prices, window_length, polyorder, deriv=2)

            df["稳健_平滑价"] = df["收盘"].ewm(span=10, adjust=False).mean()
            df["稳健_一阶导"] = df["稳健_平滑价"].diff()
            df["稳健_二阶导"] = df["稳健_一阶导"].diff()

            latest_close = df["收盘"].iloc[-1]

            # 灵敏版判断：动量+曲率组合（含波谷确认与提前信号）
            sensitive_conclusion, _ = evaluate_version_conclusion(
                sensitive_first[-2], sensitive_first[-1], sensitive_second[-2], sensitive_second[-1]
            )

            # 稳健版判断：动量+曲率组合（含波谷确认与提前信号）
            robust_first = df["稳健_一阶导"].values
            robust_second = df["稳健_二阶导"].values
            robust_conclusion, _ = evaluate_version_conclusion(
                robust_first[-2], robust_first[-1], robust_second[-2], robust_second[-1]
            )

            sensitive_buy = sensitive_conclusion == "买入"
            robust_buy = robust_conclusion == "买入"

            if sensitive_buy or robust_buy:
                if sensitive_buy and robust_buy:
                    version_label = "灵敏版+稳健版均触发"
                elif sensitive_buy:
                    version_label = "灵敏版触发"
                else:
                    version_label = "稳健版触发"

                found.append({
                    "代码": code,
                    "名称": name,
                    "最新价": round(float(latest_close), 2),
                    "触发版本": version_label
                })
        except Exception:
            continue

        time.sleep(0.3)

    return found


def get_recommendations(force_refresh=False):
    """
    获取推荐结果：优先使用3小时内的缓存，否则重新扫描
    返回：(结果列表, 是否为缓存结果, 缓存时间字符串或None)
    """
    cache = load_recommend_cache()

    if not force_refresh and is_cache_valid(cache):
        return cache["results"], True, cache["timestamp"]

    return None, False, None


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
            if data_source != "东方财富":
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

            # ===== 计算灵敏版、稳健版各自结论（不单独展示合并后的"技术面结论"）=====
            _, _, _, sensitive_ver_conclusion, robust_ver_conclusion = get_technical_conclusion(
                prev["灵敏_一阶导"], latest["灵敏_一阶导"], prev["灵敏_二阶导"], latest["灵敏_二阶导"],
                prev["稳健_一阶导"], latest["稳健_一阶导"], prev["稳健_二阶导"], latest["稳健_二阶导"]
            )

            if is_stop_loss_triggered:
                st.error("已触发止损线，建议考虑离场，控制风险")

            # ===== 资金面结论：手动触发查询，查询完成后与灵敏版、稳健版结论一起展示 =====
            st.markdown("### 查询资金面数据")
            st.caption("股东增减持公告与北向资金查询耗时较长，默认不自动执行，点击下方按钮后，将与灵敏版、稳健版技术面结论一起展示")

            capital_button = st.button("查询资金面数据（股东增减持 + 北向资金）", key=f"capital_btn_{stock_code}")

            capital_cache_key = f"capital_result_{stock_code}"

            if capital_button:
                with st.spinner("正在查询股东增减持与北向资金数据，可能需要一些时间..."):
                    capital_conclusion, capital_level, capital_details = get_capital_conclusion(stock_code)
                    st.session_state[capital_cache_key] = (capital_conclusion, capital_level, capital_details)

            if capital_cache_key in st.session_state:
                capital_conclusion, capital_level, capital_details = st.session_state[capital_cache_key]

                _, sensitive_reason_box = evaluate_version_conclusion(
                    prev["灵敏_一阶导"], latest["灵敏_一阶导"], prev["灵敏_二阶导"], latest["灵敏_二阶导"]
                )
                _, robust_reason_box = evaluate_version_conclusion(
                    prev["稳健_一阶导"], latest["稳健_一阶导"], prev["稳健_二阶导"], latest["稳健_二阶导"]
                )
                level_map_box = {"买入": "success", "卖出": "error", "观望": "info", "-": "error"}

                st.markdown("### 三项结论一览（灵敏版 / 稳健版 / 资金面）")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    render_conclusion_box("灵敏版结论（Savitzky-Golay）", sensitive_ver_conclusion, level_map_box[sensitive_ver_conclusion], sensitive_reason_box)
                with col_b:
                    render_conclusion_box("稳健版结论（EMA）", robust_ver_conclusion, level_map_box[robust_ver_conclusion], robust_reason_box)
                with col_c:
                    render_conclusion_box("资金面结论（股东+北向资金）", capital_conclusion, capital_level, "；".join(capital_details))

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
                    "灵敏版结论": "代码格式错误", "稳健版结论": "-", "成交量": "-"
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
                    "灵敏版结论": "数据获取失败", "稳健版结论": "-", "成交量": "-"
                })
                progress_bar.progress((i + 1) / len(codes))
                continue

            b_df = b_df.sort_values("日期").reset_index(drop=True)
            b_close_prices = b_df["收盘"].values

            b_df["灵敏_一阶导"] = savgol_filter(b_close_prices, window_length, polyorder, deriv=1)
            b_df["灵敏_二阶导"] = savgol_filter(b_close_prices, window_length, polyorder, deriv=2)

            b_df["稳健_平滑价"] = b_df["收盘"].ewm(span=10, adjust=False).mean()
            b_df["稳健_一阶导"] = b_df["稳健_平滑价"].diff()
            b_df["稳健_二阶导"] = b_df["稳健_一阶导"].diff()

            b_latest = b_df.iloc[-1]
            b_prev = b_df.iloc[-2]

            sensitive_result, _ = evaluate_version_conclusion(
                b_prev["灵敏_一阶导"], b_latest["灵敏_一阶导"], b_prev["灵敏_二阶导"], b_latest["灵敏_二阶导"]
            )
            robust_result, _ = evaluate_version_conclusion(
                b_prev["稳健_一阶导"], b_latest["稳健_一阶导"], b_prev["稳健_二阶导"], b_latest["稳健_二阶导"]
            )

            b_vol_msg, b_vol_level = check_volume_confirmation(b_df)
            vol_short = "放量" if "放量" in b_vol_msg else ("缩量" if ("缩量" in b_vol_msg or "萎缩" in b_vol_msg) else "平稳")

            icon_map = {"买入": "🟢 买入", "观望": "🟡 观望", "卖出": "🔴 卖出",
                        "积极": "🟢 积极", "中性": "⚪ 中性", "消极": "🔴 消极"}

            results_table.append({
                "代码": code,
                "名称": b_name if b_name else "-",
                "最新价": round(b_latest["收盘"], 2),
                "灵敏版结论": icon_map.get(sensitive_result, sensitive_result),
                "稳健版结论": icon_map.get(robust_result, robust_result),
                "成交量": vol_short
            })

            progress_bar.progress((i + 1) / len(codes))
            time.sleep(1)

        status_text.text("分析完成")
        result_df = pd.DataFrame(results_table)
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.caption("提示：批量模式仅展示核心摘要，如需查看某只股票的详细图表和止损分析，请在上方单独输入该股票代码分析")


# ========== 智能推荐 ==========

st.markdown("---")
st.markdown("## 智能推荐（低价潜力股）")
st.caption(f"从全市场A股中按价格从低到高排序扫描（优先10元、20元以内的低价股，但不排除任何价格区间），"
           f"找出动量（灵敏版或稳健版任一）出现买入信号的{RECOMMEND_TARGET_COUNT}只股票。"
           f"首批扫描约{RECOMMEND_SCAN_POOL_SIZE}只，若不够会自动扩大扫描范围，直到扫描全市场为止，确保凑够{RECOMMEND_TARGET_COUNT}只。"
           f"只有成功凑够{RECOMMEND_TARGET_COUNT}只时，结果才会缓存{RECOMMEND_VALID_HOURS}小时（有效期内重复点击直接显示缓存）；"
           f"若未凑够{RECOMMEND_TARGET_COUNT}只，下次点击将立即重新扫描，不受时间限制")

cache_check = load_recommend_cache()
if is_cache_valid(cache_check):
    cache_time_obj = datetime.fromisoformat(cache_check["timestamp"])
    remaining_minutes = int(RECOMMEND_VALID_HOURS * 60 - (datetime.now() - cache_time_obj).total_seconds() / 60)
    st.info(f"当前有缓存结果（生成于 {cache_time_obj.strftime('%Y-%m-%d %H:%M')}），约{remaining_minutes}分钟后过期")

col_r1, col_r2 = st.columns(2)
with col_r1:
    recommend_button = st.button("获取推荐（优先使用缓存）", type="primary")
with col_r2:
    force_refresh_button = st.button("强制重新扫描（忽略缓存）")

if recommend_button or force_refresh_button:
    results, is_cached, cache_timestamp = get_recommendations(force_refresh=force_refresh_button)

    if is_cached:
        st.success(f"使用缓存结果（生成于 {datetime.fromisoformat(cache_timestamp).strftime('%Y-%m-%d %H:%M')}）")
        recommend_results = results
    else:
        st.warning("正在重新扫描全市场，请耐心等待（预计5-8分钟，请勿关闭页面）...")
        progress_bar_r = st.progress(0)
        status_text_r = st.empty()

        def update_progress(current, total, found_count):
            progress_bar_r.progress(min(current / total, 1.0))
            status_text_r.text(f"已扫描 {current}/{total} 只，已找到 {found_count}/{RECOMMEND_TARGET_COUNT} 只符合条件的股票")

        recommend_results = scan_for_recommendations(progress_callback=update_progress)

        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "results": recommend_results
        }
        save_recommend_cache(cache_data)
        status_text_r.text("扫描完成")

    if not recommend_results:
        st.error("候选股票池获取失败（可能是网络波动导致全市场行情数据无法拉取），请稍等片刻后点击「强制重新扫描」重试")
    else:
        st.markdown(f"### 推荐结果（共{len(recommend_results)}只）")
        recommend_df = pd.DataFrame(recommend_results)
        st.dataframe(recommend_df, use_container_width=True, hide_index=True)
        st.caption("以上结果仅基于动量指标的历史规则计算，不构成投资建议，具体买卖请结合上方单只股票分析做进一步确认")