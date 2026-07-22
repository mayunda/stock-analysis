import akshare as ak
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ===== 第1步：设置中文字体，避免matplotlib画图时中文变成方块 =====
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows自带黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ===== 第2步：拉取股票数据 =====
stock_code = "600519"  # 贵州茅台，后面可以换成任意A股代码
df = ak.stock_zh_a_hist(
    symbol=stock_code,
    period="daily",
    start_date="20240101",
    end_date="20241231",
    adjust="qfq"
)

# 按日期排序（akshare返回的数据通常已经是按时间顺序的，这里加一道保险）
df = df.sort_values("日期").reset_index(drop=True)

# 提取收盘价这一列，转成numpy数组方便做数学计算
close_prices = df["收盘"].values

# ===== 第3步：平滑处理（关键！原始价格噪声大，直接求导没有意义） =====
# window_length: 平滑窗口大小，必须是奇数，越大越平滑，但会丢失细节
# polyorder: 拟合多项式的阶数，一般用2或3
window_length = 11
polyorder = 3

smoothed_prices = savgol_filter(close_prices, window_length, polyorder)

# ===== 第4步：计算一阶导数（价格变化速度） =====
first_derivative = savgol_filter(close_prices, window_length, polyorder, deriv=1)

# ===== 第5步：计算二阶导数（价格变化加速度） =====
second_derivative = savgol_filter(close_prices, window_length, polyorder, deriv=2)

# ===== 第6步：把结果存回DataFrame，方便查看和后续使用 =====
df["平滑收盘价"] = smoothed_prices
df["一阶导"] = first_derivative
df["二阶导"] = second_derivative

# 打印最后10天的结果看看
print(df[["日期", "收盘", "平滑收盘价", "一阶导", "二阶导"]].tail(10))

# ===== 第7步：画图，一共画3个子图，方便对比 =====
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

dates = df["日期"]

# 子图1：原始价格 vs 平滑价格
axes[0].plot(dates, close_prices, label="原始收盘价", alpha=0.5, color="gray")
axes[0].plot(dates, smoothed_prices, label="平滑后价格", color="blue")
axes[0].set_title(f"{stock_code} 收盘价（原始 vs 平滑）")
axes[0].legend()
axes[0].grid(True)

# 子图2：一阶导数
axes[1].plot(dates, first_derivative, color="green")
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")  # 画一条0基准线
axes[1].set_title("一阶导数（涨跌速度）")
axes[1].grid(True)

# 子图3：二阶导数
axes[2].plot(dates, second_derivative, color="red")
axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[2].set_title("二阶导数（涨跌加速度）")
axes[2].grid(True)

# x轴日期太密，只显示部分刻度，避免重叠看不清
step = max(len(dates) // 15, 1)
axes[2].set_xticks(dates[::step])
axes[2].set_xticklabels(dates[::step], rotation=45)

plt.tight_layout()
plt.savefig("stock_analysis.png", dpi=150)  # 保存图片到本地文件
plt.show()  # 弹窗显示图片

print("\n图片已保存为 stock_analysis.png")