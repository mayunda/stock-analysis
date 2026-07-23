import akshare as ak

df = ak.stock_info_a_code_name()
print(df.head())
print(df.columns.tolist())

# 测试查找600519对应的名称
result = df[df["code"] == "600519"]
print(result)