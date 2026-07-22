import akshare as ak

# 获取贵州茅台（股票代码 600519）的日K线数据
df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20240101", end_date="20241231", adjust="qfq")

print(df.head())
import akshare as ak

df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20240101", end_date="20241231", adjust="qfq")

print(df.head())
print(df.columns.tolist())