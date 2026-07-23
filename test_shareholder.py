import akshare as ak

print("===== 测试1：股东增减持公告 =====")
try:
    df1 = ak.stock_ggcg_em(symbol="全部")
    print(df1.columns.tolist())
    print(df1[df1["代码"] == "600519"].head())
except Exception as e:
    print(f"失败: {e}")

print("\n===== 测试2：北向资金个股持股变化 =====")
try:
    df2 = ak.stock_hsgt_individual_em(symbol="600519")
    print(df2.columns.tolist())
    print(df2.tail())
except Exception as e:
    print(f"失败: {e}")