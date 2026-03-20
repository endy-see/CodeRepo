#!/usr/bin/env python3
"""
Finance Monster — 模块1: 中国市场数据
覆盖: A股指数、个股、ETF、行业/概念板块、涨停跌停

数据源: 新浪财经(主)、同花顺(板块)、东方财富(仅涨停小数据)
⚠️ 禁止使用 *_em 批量接口！
"""
import sys
import re

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    print("❌ 请先安装依赖: pip install akshare pandas")
    sys.exit(1)


# ═══════════════════════════════════════════════════
# 股票名称 ↔ 代码映射
# ═══════════════════════════════════════════════════
_NAME_MAP = {
    "600519": "贵州茅台", "601318": "中国平安", "600036": "招商银行",
    "300750": "宁德时代", "002594": "比亚迪", "600030": "中信证券",
    "601398": "工商银行", "601939": "建设银行", "601288": "农业银行",
    "601988": "中国银行", "000002": "万科A", "000651": "格力电器",
    "000333": "美的集团", "000858": "五粮液", "603288": "海天味业",
    "603259": "药明康德", "600276": "恒瑞医药", "688981": "中芯国际",
    "600690": "海尔智家", "600900": "长江电力", "601899": "紫金矿业",
    "300760": "迈瑞医疗", "601012": "隆基绿能", "600031": "三一重工",
    "601166": "兴业银行", "600887": "伊利股份", "601857": "中国石油",
    "600028": "中国石化", "601668": "中国建筑", "000001": "平安银行",
    "600000": "浦发银行", "601088": "中国神华", "600050": "中国联通",
    "002714": "牧原股份", "300059": "东方财富", "002415": "海康威视",
    "600309": "万华化学", "002304": "洋河股份", "601728": "中国电信",
    "000568": "泸州老窖", "600809": "山西汾酒", "002230": "科大讯飞",
    "000725": "京东方A", "002475": "立讯精密",
}

_CODE_MAP = {}
for _c, _n in _NAME_MAP.items():
    _CODE_MAP[_n] = _c
    if len(_n) >= 4:
        _CODE_MAP[_n[:2]] = _c


def _get_stock_name(code: str) -> str:
    if code in _NAME_MAP:
        return _NAME_MAP[code]
    try:
        spot = ak.stock_individual_info_em(symbol=code)
        name_row = spot[spot["item"] == "股票简称"]
        return name_row.iloc[0]["value"] if not name_row.empty else code
    except Exception:
        return code


# ═══════════════════════════════════════════════════
# 查询功能
# ═══════════════════════════════════════════════════

def query_indices() -> str:
    """大盘主要指数行情（新浪日线源）"""
    targets = [
        ("sh000001", "上证指数"), ("sz399001", "深证成指"),
        ("sz399006", "创业板指"), ("sh000300", "沪深300"),
        ("sh000905", "中证500"),
    ]
    lines = ["📊 主要指数行情\n"]
    for sym, name in targets:
        try:
            df = ak.stock_zh_index_daily(symbol=sym)
            row, prev = df.iloc[-1], df.iloc[-2]
            close = float(row["close"])
            prev_close = float(prev["close"])
            pct = (close - prev_close) / prev_close * 100
            emoji = "🔴" if pct >= 0 else "🟢"
            lines.append(f"{emoji} {name}: {close:.2f} ({pct:+.2f}%)  [{row['date']}]")
        except Exception:
            lines.append(f"⚠️ {name}: 数据暂不可用")
    return "\n".join(lines)


def query_stock(code: str) -> str:
    """个股行情（新浪日线，前复权）"""
    prefix = "sh" if code.startswith("6") else "sz"
    try:
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
        if df.empty:
            return f"⚠️ 未找到 {code} 的数据"
        row, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        close = float(row["close"])
        prev_close = float(prev["close"])
        pct = (close - prev_close) / prev_close * 100 if prev_close else 0
        amt = close - prev_close
        high, low = float(row.get("high", 0)), float(row.get("low", 0))
        amp = (high - low) / prev_close * 100 if prev_close else 0
        turnover = float(row.get("turnover", 0)) * 100
        emoji = "🔴" if pct >= 0 else "🟢"
        name = _get_stock_name(code)
        return "\n".join([
            f"📈 {name} ({code})\n",
            f"  日期: {row['date']}",
            f"  收盘价: {close}",
            f"  {emoji} 涨跌幅: {pct:+.2f}%  涨跌额: {amt:+.2f}",
            f"  今开: {row.get('open', 'N/A')}  最高: {high}  最低: {low}",
            f"  成交量: {int(row.get('volume', 0)):,}  成交额: {float(row.get('amount', 0)):,.0f}",
            f"  振幅: {amp:.2f}%  换手率: {turnover:.2f}%",
        ])
    except Exception as e:
        return f"⚠️ 个股查询失败 ({code}): {str(e)[:200]}"


def query_etf(code: str) -> str:
    """ETF行情（新浪源）"""
    prefix = "sh" if code[:2] in ("51", "50", "58") else "sz"
    try:
        df = ak.fund_etf_hist_sina(symbol=f"{prefix}{code}")
        if df.empty:
            return f"⚠️ 未找到 ETF {code} 的数据"
        row = df.iloc[-1]
        close = float(row["close"])
        prev_close = float(df.iloc[-2]["close"]) if len(df) > 1 else close
        pct = (close - prev_close) / prev_close * 100 if prev_close else 0
        emoji = "🔴" if pct >= 0 else "🟢"
        return "\n".join([
            f"📈 ETF {code}\n",
            f"  日期: {row['date']}",
            f"  收盘价: {close}",
            f"  {emoji} 涨跌幅: {pct:+.2f}%",
            f"  今开: {row.get('open', 'N/A')}  最高: {row.get('high', 'N/A')}  最低: {row.get('low', 'N/A')}",
            f"  成交量: {int(row.get('volume', 0)):,}  成交额: {float(row.get('amount', 0)):,.0f}",
        ])
    except Exception as e:
        return f"⚠️ ETF 查询失败: {str(e)[:200]}"


def query_industry_boards() -> str:
    """行业板块排行（同花顺源）"""
    try:
        df = ak.stock_board_industry_summary_ths()
        lines = ["🧩 行业板块排行\n"]
        for _, row in df.head(15).iterrows():
            change = float(row.get("涨跌幅", 0))
            emoji = "🔴" if change >= 0 else "🟢"
            leader = row.get("领涨股", "")
            leader_chg = row.get("领涨股-涨跌幅", "")
            lines.append(f"  {emoji} {row['板块']}: {change:+.2f}%  领涨: {leader}({leader_chg}%)")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ 行业板块数据暂不可用: {str(e)[:100]}"


def query_concept_boards() -> str:
    """概念板块列表（同花顺源）"""
    try:
        df = ak.stock_board_concept_name_ths()
        lines = ["💡 概念板块列表 (前15)\n"]
        for _, row in df.head(15).iterrows():
            lines.append(f"  {row.get('name', row.get('板块名称', ''))} ({row.get('code', '')})")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ 概念板块数据暂不可用: {str(e)[:100]}"


def query_limit_up() -> str:
    """涨停池（东方财富小数据量接口）"""
    try:
        df = ak.stock_zt_pool_em(date=pd.Timestamp.now().strftime("%Y%m%d"))
        lines = [f"🚀 今日涨停 (共{len(df)}只)\n"]
        for _, row in df.head(15).iterrows():
            lines.append(f"  {row.get('名称','')}: {row.get('最新价','N/A')} | 连板{row.get('连板数',1)}天 | {row.get('所属行业','')}")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ 涨停数据暂不可用: {str(e)[:100]}"


# ═══════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════

def route(query: str) -> str:
    """根据自然语言查询路由到对应功能"""
    # 大盘
    if any(kw in query for kw in ["大盘", "上证", "深证", "创业板", "沪深300", "指数", "收盘"]):
        return query_indices()

    # 涨停/跌停
    if any(kw in query for kw in ["涨停", "跌停", "连板"]):
        return query_limit_up()

    # 行业板块
    if any(kw in query for kw in ["行业板块", "行业涨", "行业跌", "板块排行"]):
        return query_industry_boards()

    # 概念板块
    if any(kw in query for kw in ["概念板块", "概念涨", "概念跌"]):
        return query_concept_boards()

    # 提取6位代码
    code_match = re.search(r'\b(\d{6})\b', query)
    code = code_match.group(1) if code_match else None

    # ETF
    if code and code[:2] in ("51", "15", "16", "50", "58"):
        return query_etf(code)

    # 个股
    if code:
        return query_stock(code)

    # 中文名模糊匹配
    if re.search(r'[\u4e00-\u9fff]', query):
        clean = re.sub(r'[今日的表现怎么样最新行情走势如何分析 ]', '', query)
        for name_key, code_val in _CODE_MAP.items():
            if name_key in clean:
                return query_stock(code_val)

    return f"⚠️ 未识别的查询: '{query}'。请使用6位代码或中文名称。"


def main():
    if len(sys.argv) < 2:
        print("用法: python china_market.py <查询>\n示例: python china_market.py 大盘行情")
        sys.exit(0)
    query = " ".join(sys.argv[1:])
    if query.lower() in ["帮助", "help", "-h", "--help"]:
        print("📊 中国市场数据模块\n"
              "  大盘行情 / 600519 / 贵州茅台 / ETF 513180\n"
              "  行业板块 / 概念板块 / 今日涨停")
        sys.exit(0)
    print(route(query))


if __name__ == "__main__":
    main()
