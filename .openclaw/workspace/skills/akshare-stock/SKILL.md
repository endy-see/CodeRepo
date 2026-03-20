---
name: akshare-stock
description: A股量化数据分析工具。注意：东方财富接口(*_em)有反爬限制，容易超时断连。查询A股/ETF数据请优先使用 akshare-wrapper 技能的 main.py 脚本。
---

# A股量化 - AkShare 数据接口

> ⚠️ **重要提示**: 东方财富接口 (`*_em`) 当前被反爬封禁，会导致 `RemoteDisconnected` 超时。
> **请勿使用** `stock_zh_a_spot_em()`、`stock_zh_a_hist()`、`stock_board_*_em()`、`fund_etf_*_em()` 等接口。
> **推荐做法**: 直接执行 akshare-wrapper 脚本：
> ```bash
> exec python ~/.openclaw/workspace/skills/akshare-wrapper/main.py "A股大盘"
> exec python ~/.openclaw/workspace/skills/akshare-wrapper/main.py "600519 今日表现"
> exec python ~/.openclaw/workspace/skills/akshare-wrapper/main.py "513180 今日表现"
> ```

## 可用的稳定数据源（新浪/同花顺）

### 指数数据（新浪源）
```python
import akshare as ak
# 指数历史日线（新浪源，稳定）
df = ak.stock_zh_index_daily(symbol="sh000001")  # 上证指数
df = ak.stock_zh_index_daily(symbol="sz399001")  # 深证成指
```

### 个股历史（新浪源）
```python
import akshare as ak
# 个股日线（新浪源，稳定）
df = ak.stock_zh_a_daily(symbol="sh600519", adjust="qfq")  # 贵州茅台
df = ak.stock_zh_a_daily(symbol="sz000001", adjust="qfq")  # 平安银行
# 交易所前缀: 6开头->sh, 0/3开头->sz
```

### ETF数据（新浪源）
```python
import akshare as ak
# ETF历史（新浪源，稳定）
df = ak.fund_etf_hist_sina(symbol="sh513180")  # 恒生科技ETF
df = ak.fund_etf_hist_sina(symbol="sz159919")  # 沪深300ETF
# 51/50/58 开头->sh, 15/16 开头->sz
```

### 行业板块（同花顺源）
```python
import akshare as ak
df = ak.stock_board_industry_summary_ths()  # 行业板块排行（含涨跌幅）
df = ak.stock_board_industry_name_ths()     # 行业板块列表
```

## ⛔ 以下接口当前不可用（东方财富反爬封禁）

以下接口会导致 `RemoteDisconnected` 超时，**请勿使用**：
- `stock_zh_a_spot_em()` — 全市场实时行情
- `stock_zh_a_hist()` — 个股历史K线
- `stock_board_industry_name_em()` — 行业板块
- `stock_board_concept_name_em()` — 概念板块
- `fund_etf_hist_em()` / `fund_etf_spot_em()` — ETF数据
- `index_zh_a_hist()` — 指数历史

请使用上方的新浪/同花顺替代接口，或直接调用 akshare-wrapper 脚本。

## 常用股票代码

- **平安银行**: 000001
- **贵州茅台**: 600519
- **宁德时代**: 300750
- **比亚迪**: 002594
- **招商银行**: 600036

## 备选方案: Baostock

如果 AkShare 安装失败，可使用 baostock（更轻量）:

```python
import baostock as bs

# 登录
lg = bs.login()
print(lg.error_msg)

# 获取历史K线
rs = bs.query_history_k_data_plus('sh.600519',
    'date,code,open,high,low,close,volume',
    start_date='20250101',
    end_date='20251231')

data_list = []
while rs.next:
    data_list.append(rs.get_row_data())
    
bs.logout()
```

## ETF基金查询

> ⚠️ **ETF代码（51/15/16/50/58开头）不在 `stock_zh_a_spot_em()` 结果中**，请用以下方式查询：

```python
import akshare as ak

# ETF历史行情（推荐使用新浪源，稳定不会超时）
# 上交所 ETF (51/50/58 开头) 加 "sh" 前缀
df = ak.fund_etf_hist_sina(symbol="sh513180")  # 恒生科技ETF
df = ak.fund_etf_hist_sina(symbol="sh510050")  # 上证50ETF

# 深交所 ETF (15/16 开头) 加 "sz" 前缀
df = ak.fund_etf_hist_sina(symbol="sz159919")  # 沪深300ETF
```

> ⚠️ **不要使用** `fund_etf_hist_em()` 或 `fund_etf_spot_em()`，东方财富接口有反爬限制，经常超时。

**更推荐**: 直接调用 `akshare-wrapper` 技能的脚本：
```bash
exec python ~/.openclaw/workspace/skills/akshare-wrapper/main.py "513180 今日表现"
```

## 注意事项

1. 数据仅供学术研究，不构成投资建议
2. 接口可能因目标网站变动而失效
3. 建议添加异常处理和重试机制
4. **东方财富接口 (`*_em`) 有反爬机制，全量拉取容易超时建议用 `*_sina` 替代或使用 akshare-wrapper 脚本**
5. ETF代码不在A股接口中，需单独使用 `fund_etf_hist_sina()` 查询
