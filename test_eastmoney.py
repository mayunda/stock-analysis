import requests

url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
params = {
    "secid": "1.600519",
    "fields1": "f1,f2,f3,f4,f5",
    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
    "klt": "101",
    "fqt": "1",
    "beg": "20240101",
    "end": "20241231"
}
headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, params=params, headers=headers, timeout=10)
print("状态码:", response.status_code)
print(response.text[:200])