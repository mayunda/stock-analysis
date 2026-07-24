import akshare as ak

print("===== 测试1：新浪全市场快照（作为全市场候选池的备用源）=====")
try:
    df1 = ak.stock_zh_a_spot()
    print(df1.columns.tolist())
    print(df1.head())
except Exception as e:
    print(f"失败: {e}")

print("\n===== 测试2：新浪个股历史K线（作为历史数据的第三备用源）=====")
try:
    df2 = ak.stock_zh_a_daily(symbol="sh600519")
    print(df2.columns.tolist())
    print(df2.tail())
except Exception as e:
    print(f"失败: {e}")