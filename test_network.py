import requests

response = requests.get("https://www.baidu.com", timeout=10)
print("状态码:", response.status_code)