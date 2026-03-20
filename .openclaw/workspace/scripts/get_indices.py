import akshare as ak
import pandas as pd

df = ak.stock_zh_index_spot_em()
indices = ["上证指数", "深证成指", "创业板指", "沪深300", "中证500"]
df = df[df["名称"].isin(indices)]
print("📈 A股主要指数收盘行情 (2026-03-19)：")
for _, row in df.iterrows():
    change = float(row.get("涨跌幅", 0))
    emoji = "🔴" if change >= 0 else "🟢"
    print(f"{emoji} {row['名称']}: {row.get('最新价', 'N/A')} ({change:+.2f}%)")
