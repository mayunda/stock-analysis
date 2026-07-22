import akshare as ak

print("测试1：尝试新浪财经接口...")
try:
    df1 = ak.stock_zh_a_hist_tx(symbol="sh600519")
    print("成功！最近数据：")
    print(df1.tail())
except Exception as e:
    print(f"失败: {e}")

print("\n测试2：尝试实时行情接口...")
try:
    df2 = ak.stock_zh_a_spot_em()
    print("成功！数据量：", len(df2))
except Exception as e:
    print(f"失败: {e}")