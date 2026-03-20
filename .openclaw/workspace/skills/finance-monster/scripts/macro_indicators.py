#!/usr/bin/env python3
"""
Finance Monster — 模块6: 宏观市场指标
覆盖: 10年期/30年期国债收益率（中国 & 美国）

数据源: akshare bond_zh_us_rate (新浪/同花顺)
⚠️ 禁止使用 *_em 批量接口！
"""
import sys
import os
import re
from datetime import datetime, timedelta

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    print("❌ 请先安装依赖: pip install akshare pandas")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, AutoDateLocator
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# 中文字体配置
if _HAS_MPL:
    for _font in ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei"]:
        try:
            matplotlib.rcParams["font.sans-serif"] = [_font] + matplotlib.rcParams["font.sans-serif"]
            break
        except Exception:
            pass
    matplotlib.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════
# 国债收益率查询
# ═══════════════════════════════════════════════════

def query_bond_yield(period: str = "all", days: int = 365, chart: bool = True) -> str:
    """
    查询中国/美国国债收益率。

    Args:
        period: "10y", "30y", "all" — 查询哪个期限
        days: 回溯天数，默认365天（最近一年）
        chart: 是否生成走势图
    Returns:
        格式化的文本输出
    """
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    try:
        df = ak.bond_zh_us_rate(start_date=start)
    except Exception as e:
        return f"❌ 获取国债收益率失败: {e}"

    if df is None or df.empty:
        return "⚠️ 未获取到国债收益率数据"

    # 标准列名
    col_date = "日期"
    col_cn10 = "中国国债收益率10年"
    col_cn30 = "中国国债收益率30年"
    col_us10 = "美国国债收益率10年"
    col_us30 = "美国国债收益率30年"

    # 选列
    if period == "10y":
        cols = [col_date, col_cn10, col_us10]
        title = "10年期国债收益率"
    elif period == "30y":
        cols = [col_date, col_cn30, col_us30]
        title = "30年期国债收益率"
    else:
        cols = [col_date, col_cn10, col_cn30, col_us10, col_us30]
        title = "10年期 & 30年期国债收益率"

    # 过滤有效列
    available = [c for c in cols if c in df.columns]
    if len(available) <= 1:
        return f"⚠️ 数据中缺少所需列。可用列: {df.columns.tolist()}"

    result = df[available].copy()
    data_cols = [c for c in available if c != col_date]
    result = result.dropna(subset=data_cols, how="all")

    if result.empty:
        return "⚠️ 指定时间范围内无有效数据"

    result[col_date] = pd.to_datetime(result[col_date])
    result = result.sort_values(col_date)

    # 统计信息
    lines = [f"📈 {title}（最近 {days} 天）\n"]
    lines.append(f"数据范围: {result[col_date].iloc[0].strftime('%Y-%m-%d')} ~ "
                 f"{result[col_date].iloc[-1].strftime('%Y-%m-%d')}  "
                 f"共 {len(result)} 条\n")

    # 最近5条明细
    recent = result.tail(5)
    lines.append("--- 最近 5 条 ---")
    for _, row in recent.iterrows():
        date_str = row[col_date].strftime("%Y-%m-%d")
        vals = []
        for c in data_cols:
            v = row[c]
            label = c.replace("国债收益率", "")
            if pd.notna(v):
                vals.append(f"{label}: {float(v):.4f}%")
            else:
                vals.append(f"{label}: --")
        lines.append(f"  {date_str}  {' | '.join(vals)}")

    # 汇总统计
    lines.append("\n--- 统计摘要 ---")
    for c in data_cols:
        series = pd.to_numeric(result[c], errors="coerce").dropna()
        if series.empty:
            continue
        label = c.replace("国债收益率", "")
        lines.append(
            f"  {label}: "
            f"最新={series.iloc[-1]:.4f}%  "
            f"均值={series.mean():.4f}%  "
            f"最高={series.max():.4f}%({result.loc[series.idxmax(), col_date].strftime('%m-%d')})  "
            f"最低={series.min():.4f}%({result.loc[series.idxmin(), col_date].strftime('%m-%d')})"
        )

    # 中美利差
    if col_cn10 in available and col_us10 in available:
        cn10 = pd.to_numeric(result[col_cn10], errors="coerce")
        us10 = pd.to_numeric(result[col_us10], errors="coerce")
        spread = (cn10 - us10).dropna()
        if not spread.empty:
            lines.append(f"\n  🔀 中美10年利差(最新): {spread.iloc[-1]:.4f}%")

    if col_cn30 in available and col_us30 in available:
        cn30 = pd.to_numeric(result[col_cn30], errors="coerce")
        us30 = pd.to_numeric(result[col_us30], errors="coerce")
        spread = (cn30 - us30).dropna()
        if not spread.empty:
            lines.append(f"  🔀 中美30年利差(最新): {spread.iloc[-1]:.4f}%")

    # 生成走势图
    if chart:
        chart_path = plot_bond_yield(result, col_date, data_cols, title, days)
        if chart_path:
            lines.append(f"\n📊 走势图已保存: {chart_path}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════

# 配色方案
_COLORS = {
    "中国国债收益率10年": "#E63946",
    "中国国债收益率30年": "#D62828",
    "美国国债收益率10年": "#457B9D",
    "美国国债收益率30年": "#1D3557",
}
_LABELS = {
    "中国国债收益率10年": "中国10年",
    "中国国债收益率30年": "中国30年",
    "美国国债收益率10年": "美国10年",
    "美国国债收益率30年": "美国30年",
}


def plot_bond_yield(df: pd.DataFrame, col_date: str, data_cols: list,
                    title: str, days: int) -> str:
    """
    绘制国债收益率走势图并保存为PNG。

    Returns:
        图片文件绝对路径，失败返回空字符串
    """
    if not _HAS_MPL:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))

    dates = df[col_date]
    for col in data_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        color = _COLORS.get(col, None)
        label = _LABELS.get(col, col.replace("国债收益率", ""))
        # 粗线=中国，细线=美国
        lw = 2.2 if "中国" in col else 1.6
        ls = "-" if "中国" in col else "--"
        ax.plot(dates, series, label=label, color=color, linewidth=lw, linestyle=ls)

        # 标注最新值
        last_valid = series.dropna()
        if not last_valid.empty:
            last_idx = last_valid.index[-1]
            last_val = last_valid.iloc[-1]
            last_date = df.loc[last_idx, col_date]
            ax.annotate(f"{last_val:.2f}%",
                        xy=(last_date, last_val),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=color,
                        va="center")

    ax.set_title(f"{title}（最近 {days} 天走势）", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("收益率 (%)", fontsize=12)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=12))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()

    # 保存到 cache 目录
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"bond_yield_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(cache_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath


# ═══════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════

_PERIOD_PATTERNS = {
    "10y": re.compile(r"10年|十年|10-?year|10Y", re.IGNORECASE),
    "30y": re.compile(r"30年|三十年|30-?year|30Y", re.IGNORECASE),
}

_DAYS_PATTERNS = [
    (re.compile(r"最近(\d+)天"), lambda m: int(m.group(1))),
    (re.compile(r"最近(\d+)个?月"), lambda m: int(m.group(1)) * 30),
    (re.compile(r"最近(\d+)年|近(\d+)年"), lambda m: int(m.group(1) or m.group(2)) * 365),
    (re.compile(r"半年"), lambda _: 183),
    (re.compile(r"一年|1年"), lambda _: 365),
    (re.compile(r"两年|2年"), lambda _: 730),
    (re.compile(r"三年|3年"), lambda _: 1095),
]


def route(query: str) -> str:
    """根据用户查询分发到对应函数"""
    # 判断期限
    has_10 = bool(_PERIOD_PATTERNS["10y"].search(query))
    has_30 = bool(_PERIOD_PATTERNS["30y"].search(query))

    if has_10 and has_30:
        period = "all"
    elif has_10:
        period = "10y"
    elif has_30:
        period = "30y"
    else:
        period = "all"

    # 判断时间范围
    days = 365  # 默认一年
    for pat, fn in _DAYS_PATTERNS:
        m = pat.search(query)
        if m:
            days = fn(m)
            break

    # 判断是否需要图表
    want_chart = bool(re.search(r"图|chart|plot|走势|趋势|可视化|visual", query, re.IGNORECASE))

    return query_bond_yield(period=period, days=days, chart=want_chart)


def main():
    if len(sys.argv) < 2:
        print("用法: python macro_indicators.py <查询> [--chart]\n"
              "示例:\n"
              "  python macro_indicators.py '最近一年10年期国债收益率'\n"
              "  python macro_indicators.py '30年期国债收益率' --chart\n"
              "  python macro_indicators.py '最近半年国债收益率走势图'\n"
              "  python macro_indicators.py '中国最近3个月10年和30年国债收益率'")
        sys.exit(0)

    # 支持 --chart 标志强制生成图表
    args = sys.argv[1:]
    force_chart = "--chart" in args
    if force_chart:
        args.remove("--chart")

    query = " ".join(args)
    if query.lower() in ["帮助", "help", "-h", "--help"]:
        print("📈 宏观市场指标模块\n"
              "  支持查询: 10年期/30年期国债收益率（中国 & 美国）\n"
              "  时间范围: 最近N天/N月/N年/半年/一年\n"
              "  加 --chart 或在查询中含'走势图/趋势/可视化'自动生成图表\n"
              "  示例: '最近一年30年期国债收益率走势图'")
        sys.exit(0)

    if force_chart:
        query += " 走势图"  # 注入图表关键词，让 route() 识别
    print(route(query))


if __name__ == "__main__":
    main()
