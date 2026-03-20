---
name: finance-monster
description: |
  使用此技能当用户问到：股票、A股、ETF、基金、期货、期权、加密货币、投资组合、
  国债、收益率、利率、利差、10年期、30年期、大盘、涨停、跌停、板块、行业、
  美股、港股、BTC、行情、走势、K线、技术指标、MACD、RSI、中美利差、
  bond yield、treasury、stock、crypto、futures、options。
  本技能包含可执行的Python脚本，直接exec运行即可获取实时数据，无需上网搜索。
  所有金融/宏观数据问题都必须使用此技能的本地脚本，禁止从网上爬数据。
version: 1.0.0
---

# 🐉 Finance Monster — 终极金融分析技能

> **唯一金融技能**：所有关于股票、基金、ETF、期货、期权、加密货币、投资组合的问题一律使用本技能。

> **⚠️ 核心原则：所有金融数据必须通过本地脚本获取，严禁自行从网上抓取！**
> 每个模块都有对应的 Python 脚本，用 `exec python scripts/xxx.py` 调用即可获得数据。
> 不要尝试访问 Bloomberg、TradingEconomics、FRED、Chinamoney 等外部网站，本地 akshare/yfinance 数据源已全面覆盖。

---

## 模块架构

本技能由 6 大模块组成：

### 模块 1: 🇨🇳 中国市场数据 (`china_market.py`)
A股、ETF、指数、行业/概念板块、涨停跌停

**支持查询：**
- 大盘行情（上证/深证/创业板/沪深300/中证500）
- 个股行情（6位代码或中文名称，如 `600519` 或 `贵州茅台`）
- ETF 行情（51/15/16/50/58开头代码，如 `513180`）
- 行业板块排行（同花顺数据源）
- 概念板块列表（同花顺数据源）
- 涨停/跌停/连板统计

**数据源：** 新浪财经（主力）、同花顺（板块）、东方财富（仅涨停小数据）
**⚠️ 禁止使用任何 `*_em` 批量接口！** 东方财富已全面反爬封禁。

**调用方式：**
```bash
python scripts/china_market.py "大盘行情"
python scripts/china_market.py "600519"
python scripts/china_market.py "贵州茅台"
python scripts/china_market.py "ETF 513180"
python scripts/china_market.py "行业板块"
python scripts/china_market.py "概念板块"
python scripts/china_market.py "今日涨停"
```

---

### 模块 2: 📊 中国衍生品 (`china_derivatives.py`)
期货实时盘面、期货分钟K线+技术指标、期权IV/Greeks、期权RR25

**支持查询：**
- 期货实时行情面板（按品种查所有合约）
- 期货分钟K线 + MA/EMA/MACD/RSI 指标
- 期权隐含波动率 + Greeks（50ETF/300ETF 期权）
- 期权 RR25 风险逆转指标

**数据源：** 新浪财经

**调用方式：**
```bash
python scripts/china_derivatives.py futures-board --symbol PTA
python scripts/china_derivatives.py futures-indicators --contract IF2603 --period 5
python scripts/china_derivatives.py options-greeks --underlying 510050
python scripts/china_derivatives.py options-rr25 --underlying 510050
```

---

### 模块 3: 🌍 全球市场分析 (`global_market.py`)
美股/港股/加密货币 8维度评分分析、股息分析

**支持查询：**
- 美股深度分析（8维评分：盈利惊喜/基本面/分析师情绪/历史模式/市场环境/行业对比/动量/情绪）
- 加密货币分析（BTC关联、动量指标）
- 股息安全性/成长性分析
- 公司概况/SEC文件/内部人交易

**依赖：** yfinance, fear-and-greed, edgartools, feedparser

**调用方式：**
```bash
python scripts/global_market.py analyze AAPL
python scripts/global_market.py analyze AAPL --fast
python scripts/global_market.py analyze BTC-USD
python scripts/global_market.py dividends AAPL
```

---

### 模块 4: 📋 投资组合与监控 (`portfolio_tools.py`)
投资组合管理、自选股监控、热点扫描、传闻扫描

**支持查询：**
- 投资组合 CRUD（添加/删除/查看/盈亏追踪）
- 自选股监控 + 价格/信号警报
- 热点扫描（病毒式传播检测）
- 传闻扫描（并购/内幕/早期信号）

**调用方式：**
```bash
python scripts/portfolio_tools.py portfolio list
python scripts/portfolio_tools.py portfolio add AAPL 150 10
python scripts/portfolio_tools.py watchlist add AAPL
python scripts/portfolio_tools.py watchlist check
python scripts/portfolio_tools.py hot-scan
python scripts/portfolio_tools.py rumor-scan
```

---

### 模块 6: 📈 宏观市场指标 (`macro_indicators.py`)
10年期/30年期国债收益率（中国 & 美国），中美利差

**支持查询：**
- 10年期国债收益率（中国 & 美国）
- 30年期国债收益率（中国 & 美国）
- 10年+30年同时查询
- 中美利差计算
- 自定义时间范围（最近N天/N月/N年/半年）
- **走势图可视化**（查询含"走势图/趋势/可视化"或加 `--chart` 自动生成PNG）

**数据源：** akshare `bond_zh_us_rate`（新浪/同花顺）

**调用方式：**
```bash
python scripts/macro_indicators.py "最近一年10年期国债收益率"
python scripts/macro_indicators.py "30年期国债收益率" --chart
python scripts/macro_indicators.py "最近半年国债收益率走势图"
python scripts/macro_indicators.py "中国最近3个月10年和30年国债收益率"
```

---

### 模块 5: 🔍 研究方法论（内置规则，无脚本）
投资研究的质量控制和信息源验证

**核心规则：**
1. **官方信源优先序** — 交易所公告 > 公司IR > 监管文件 > 财经终端 > 媒体
2. **中国市场** — SSE/SZSE/BSE 公告、巨潮资讯、公司年报/半年报/季报
3. **美国市场** — SEC EDGAR 10-K/10-Q/8-K、公司IR网站
4. **基金/ETF** — 基金公司官网、招募说明书、定期报告
5. **一致性检查** — Ticker是否正确? 数据是否最新? 审计/未审计? 可比口径?

**红线（禁止作为唯一来源）：** 社交媒体、券商营销材料、未经验证的AI摘要

详见 `references/source-map.md` 和 `references/research-methodology.md`

---

## 路由规则

Agent 在收到金融相关问题时，**必须执行对应的本地 Python 脚本**，不要自行从互联网抓取数据。

**执行方式：** `exec python ~/.openclaw/workspace/skills/finance-monster/scripts/<脚本> <参数>`

| 用户意图 | 模块 | 脚本 |
|---|---|---|
| A股/指数/ETF/板块行情 | 模块1 | `china_market.py` |
| 期货/期权相关 | 模块2 | `china_derivatives.py` |
| 美股/港股/加密货币分析 | 模块3 | `global_market.py` |
| 投资组合/自选股/热点/传闻 | 模块4 | `portfolio_tools.py` |
| 投资研究方法论/信源验证 | 模块5 | 参考 references/ |
| **国债收益率/利率/中美利差/宏观指标** | **模块6** | **`macro_indicators.py`** |

### 模块6 路由关键词
以下关键词命中时，**必须**执行 `macro_indicators.py`：
- 国债、收益率、利率、10年期、30年期、十年期、三十年期
- 中美利差、treasury yield、bond yield
- 宏观指标、利率走势

## 关键参考数据

- `references/china-stocks.md` — A股/港股/美股常用代码速查表
- `references/source-map.md` — 各市场官方信息源映射
- `references/research-methodology.md` — 投资研究方法论
- `maps/keywords.yml` — 中文关键词归一化（期货/期权路由用）
- `maps/router.yml` — 衍生品路由表

## 注意事项

1. **东方财富 `*_em` 系列接口全部被封**，严禁使用！已全部替换为新浪/同花顺源
2. A股数据为**前一交易日收盘价**（非实时盘中），有 T+0 晚间延迟
3. 美股通过 Yahoo Finance 获取，有 15-20 分钟延迟
4. 名称→代码映射有限（~40只热门股），不在映射表中的请用6位代码
5. 期权数据仅覆盖 SSE 的 50ETF/300ETF 期权
