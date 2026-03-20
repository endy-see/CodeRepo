#!/usr/bin/env python3
"""
akshare-stock技能包装器
直接调用 akshare 库，无需外部依赖
"""

import sys
import os
import re
import json
from typing import Optional, Tuple

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    print("❌ 请先安装依赖: pip install akshare pandas")
    sys.exit(1)


# 常用股票名称映射（避免调用东方财富接口获取名称）
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
}

# 反向映射：名称 → 代码
_CODE_MAP = {}
for _c, _n in _NAME_MAP.items():
    _CODE_MAP[_n] = _c
    # 简称映射
    if len(_n) >= 4:
        _CODE_MAP[_n[:2]] = _c


def _get_stock_name(code: str) -> str:
    """获取股票名称，优先本地映射，fallback 调用单只接口"""
    if code in _NAME_MAP:
        return _NAME_MAP[code]
    try:
        spot = ak.stock_individual_info_em(symbol=code)
        name_row = spot[spot["item"] == "股票简称"]
        return name_row.iloc[0]["value"] if not name_row.empty else code
    except Exception:
        return code


def run_akshare_query(query: str, platform: str = "qq") -> str:
    """执行akshare查询 - 直接调用 akshare 库
    所有数据源优先使用新浪接口，避免东方财富反爬超时。

    Args:
        query: 自然语言查询
        platform: 输出平台格式 (qq/telegram)

    Returns:
        格式化后的查询结果
    """
    try:
        query_lower = query.lower()

        # ── 大盘行情（新浪日线源，稳定可靠）──
        if any(kw in query for kw in ["大盘", "上证", "深证", "创业板", "沪深300", "指数", "收盘"]):
            target_indices = [
                ("sh000001", "上证指数"),
                ("sz399001", "深证成指"),
                ("sz399006", "创业板指"),
                ("sh000300", "沪深300"),
                ("sh000905", "中证500"),
            ]
            lines = ["📊 主要指数行情\n"]
            for sym, name in target_indices:
                try:
                    df = ak.stock_zh_index_daily(symbol=sym)
                    row = df.iloc[-1]
                    prev = df.iloc[-2]
                    close = float(row["close"])
                    prev_close = float(prev["close"])
                    change_pct = (close - prev_close) / prev_close * 100
                    emoji = "🔴" if change_pct >= 0 else "🟢"
                    lines.append(f"{emoji} {name}: {close:.2f} ({change_pct:+.2f}%)  [{row['date']}]")
                except Exception:
                    lines.append(f"⚠️ {name}: 数据暂不可用")
            return "\n".join(lines)

        # ── 涨停/跌停（东方财富小数据量，可用）──
        if any(kw in query for kw in ["涨停", "跌停", "连板"]):
            df = ak.stock_zt_pool_em(date=pd.Timestamp.now().strftime("%Y%m%d"))
            lines = [f"🚀 今日涨停 (共{len(df)}只)\n"]
            for _, row in df.head(15).iterrows():
                lines.append(f"  {row.get('名称','')}: {row.get('最新价','N/A')} | 连板{row.get('连板数',1)}天 | {row.get('所属行业','')}")
            return "\n".join(lines)

        # ── 行业板块（同花顺源）──
        if any(kw in query for kw in ["行业板块", "行业涨", "行业跌", "板块"]):
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

        # ── 概念板块（同花顺源）──
        if any(kw in query for kw in ["概念板块", "概念涨", "概念跌"]):
            try:
                df = ak.stock_board_concept_name_ths()
                lines = ["💡 概念板块列表 (前15)\n"]
                for _, row in df.head(15).iterrows():
                    lines.append(f"  {row.get('name', row.get('板块名称', ''))} ({row.get('code', '')})")
                return "\n".join(lines)
            except Exception as e:
                return f"⚠️ 概念板块数据暂不可用: {str(e)[:100]}"

        # ── 提取纯数字代码 ──
        code_match = re.search(r'\b(\d{6})\b', query)
        code = code_match.group(1) if code_match else None

        # ── ETF/基金查询（新浪源）── 51/15/16/50/58 开头
        if code and code[:2] in ("51", "15", "16", "50", "58"):
            try:
                prefix = "sh" if code[:2] in ("51", "50", "58") else "sz"
                df = ak.fund_etf_hist_sina(symbol=f"{prefix}{code}")
                if df.empty:
                    return f"⚠️ 未找到 ETF {code} 的数据"
                row = df.iloc[-1]
                close = float(row["close"])
                prev_close = float(df.iloc[-2]["close"]) if len(df) > 1 else close
                change_pct = (close - prev_close) / prev_close * 100 if prev_close else 0
                emoji = "🔴" if change_pct >= 0 else "🟢"
                lines = [
                    f"📈 ETF {code}\n",
                    f"  日期: {row['date']}",
                    f"  收盘价: {close}",
                    f"  {emoji} 涨跌幅: {change_pct:+.2f}%",
                    f"  今开: {row.get('open', 'N/A')}  最高: {row.get('high', 'N/A')}  最低: {row.get('low', 'N/A')}",
                    f"  成交量: {int(row.get('volume', 0)):,}  成交额: {float(row.get('amount', 0)):,.0f}",
                ]
                return "\n".join(lines)
            except Exception as e:
                return f"⚠️ ETF 查询失败: {str(e)[:200]}"

        # ── 个股查询（A 股）── 使用新浪源 stock_zh_a_daily 避免东方财富封禁
        if code:
            try:
                # 判断交易所前缀: 6开头 -> 上交所sh, 0/3开头 -> 深交所sz, 68开头 -> 科创板sh
                if code.startswith("6"):
                    sina_sym = f"sh{code}"
                elif code.startswith(("0", "3")):
                    sina_sym = f"sz{code}"
                else:
                    sina_sym = f"sh{code}"
                df = ak.stock_zh_a_daily(symbol=sina_sym, adjust="qfq")
                if not df.empty:
                    row = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else row
                    close = float(row["close"])
                    prev_close = float(prev["close"])
                    change_pct = (close - prev_close) / prev_close * 100 if prev_close else 0
                    change_amt = close - prev_close
                    high = float(row.get("high", 0))
                    low = float(row.get("low", 0))
                    amplitude = (high - low) / prev_close * 100 if prev_close else 0
                    turnover = float(row.get("turnover", 0)) * 100  # fraction → %
                    emoji = "🔴" if change_pct >= 0 else "🟢"
                    # 获取名称（尝试简单映射，不调用东方财富）
                    name = _get_stock_name(code)
                    lines = [
                        f"📈 {name} ({code})\n",
                        f"  日期: {row['date']}",
                        f"  收盘价: {close}",
                        f"  {emoji} 涨跌幅: {change_pct:+.2f}%  涨跌额: {change_amt:+.2f}",
                        f"  今开: {row.get('open', 'N/A')}  最高: {high}  最低: {low}",
                        f"  成交量: {int(row.get('volume', 0)):,}  成交额: {float(row.get('amount', 0)):,.0f}",
                        f"  振幅: {amplitude:.2f}%  换手率: {turnover:.2f}%",
                    ]
                    return "\n".join(lines)
            except Exception as e:
                return f"⚠️ 个股查询失败 ({code}): {str(e)[:150]}"

        # ── 按名称模糊搜索 ──
        # 使用本地映射，避免全市场拉取
        if re.search(r'[\u4e00-\u9fff]', query):
            clean = re.sub(r'[今日的表现怎么样最新行情走势 ]', '', query)
            matched_code = None
            for name_key, code_val in _CODE_MAP.items():
                if name_key in clean:
                    matched_code = code_val
                    break
            if matched_code:
                return run_akshare_query(matched_code, platform)
            return f"⚠️ 未找到匹配 '{query}' 的股票。请直接使用6位代码查询，如 '600519 今日表现'"

        return f"⚠️ 未找到匹配 '{query}' 的股票或功能"

    except Exception as e:
        return f"⚠️ 查询失败: {str(e)[:200]}"

def preprocess_query(query: str) -> Tuple[str, str]:
    """预处理查询，提取平台偏好
    
    Returns:
        (processed_query, platform)
    """
    # 默认平台
    platform = "qq"
    
    # 清理查询
    query = query.strip()
    
    # 检查平台提示（未来扩展）
    if "--telegram" in query:
        platform = "telegram"
        query = query.replace("--telegram", "").strip()
    elif "--qq" in query:
        platform = "qq"
        query = query.replace("--qq", "").strip()
    
    return query, platform

def show_help() -> str:
    """显示帮助信息"""
    help_text = """📈 A股分析技能使用帮助

常用查询示例：
--------------------
📊 大盘行情
  • A股大盘
  • 上证指数
  • 创业板指

📈 个股分析
  • 贵州茅台近30日K线
  • 茅台资金流向
  • 600519怎么样

🧩 板块分析
  • 行业板块涨跌
  • 概念板块涨跌
  • 板块资金流向

🚦 市场统计
  • 今日涨停
  • 连板梯队

🌏 其他市场
  • 港股行情
  • 美股行情

📰 新闻研究
  • 财经新闻
  • 宁德时代研报

💡 使用提示：
1. 支持股票代码（600519）和常用名称（茅台）
2. 可指定时间范围（近30日、周线、月线）
3. 复杂查询可能需要更长时间
4. 数据非实时，有少许延迟

输入任意查询即可开始使用！"""
    return help_text

def main():
    """主函数：OpenClaw技能入口"""
    if len(sys.argv) < 2:
        # 没有参数时显示帮助
        print(show_help())
        sys.exit(0)
    
    # 获取查询参数
    query = " ".join(sys.argv[1:])
    
    # 处理帮助请求
    if query.lower() in ["帮助", "help", "--help", "-h", "用法", "怎么用"]:
        print(show_help())
        sys.exit(0)
    
    # 处理版本请求
    if query.lower() in ["版本", "version", "--version", "-v"]:
        print("akshare-wrapper v1.0 (2026-03-11)")
        sys.exit(0)
    
    # 预处理查询
    processed_query, platform = preprocess_query(query)
    
    # 执行查询
    result = run_akshare_query(processed_query, platform)
    
    # 输出结果
    print(result)

if __name__ == "__main__":
    main()