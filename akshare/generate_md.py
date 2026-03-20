"""
Generate akshare_data.md from test results
"""
import json
from collections import defaultdict
from datetime import datetime

INPUT = "akshare_test_results.json"
OUTPUT = "akshare_data.md"

# Category mapping by function name prefix
CATEGORIES = {
    "股票数据": ["stock_"],
    "指数数据": ["index_"],
    "期货数据": ["futures_", "future_"],
    "期权数据": ["option_"],
    "债券数据": ["bond_"],
    "外汇数据": ["fx_"],
    "货币数据": ["currency_"],
    "基金数据": ["fund_"],
    "宏观数据": ["macro_"],
    "利率数据": ["rate_"],
    "现货数据": ["spot_"],
    "能源数据": ["energy_"],
    "加密货币": ["crypto_"],
    "新闻资讯": ["news_", "stock_news"],
    "申万指数": ["sw_index", "sw_"],
    "工具箱": ["tool_"],
    "另类数据": ["air_", "car_", "movie_", "tv_", "online_", "sunrise_", "wealth_", "forbes_", "hurun_", "xincaifu_", "bloomberg_", "weibo_", "migration_"],
    "波动率/学术": ["article_"],
    "银行数据": ["bank_"],
    "中证指数": ["index_cni", "index_csindex"],
    "龙虎榜": ["stock_lhb_", "stock_sina_lhb_"],
    "涨停板": ["stock_zt_", "stock_limit_"],
    "融资融券": ["stock_margin_"],
    "沪深港通": ["stock_hsgt_", "stock_hk_ggt_"],
    "板块数据": ["stock_board_"],
    "财务报表": ["stock_balance_", "stock_profit_", "stock_cash_", "stock_financial_"],
    "分红配送": ["stock_fhps_", "stock_dividents_", "stock_dividend_"],
    "股东数据": ["stock_gdfx_", "stock_gdhs_"],
    "概念板块": ["stock_board_concept_"],
    "行业板块": ["stock_board_industry_"],
    "技术指标": ["stock_rank_"],
    "ESG": ["stock_esg_"],
    "资金流向": ["stock_fund_flow_", "stock_individual_fund_flow_", "stock_market_fund_flow_", "stock_sector_fund_flow_"],
    "商品数据": ["futures_comm_", "futures_inventory_", "futures_warehouse_"],
    "中债指数": ["bond_composite_", "bond_new_composite_"],
    "可转债": ["bond_cb_", "bond_cov_", "bond_zh_cov"],
    "REITs": ["fund_reits_"],
    "日历": ["tool_trade_date"],
    "财新指数": ["index_cx_"],
    "新股数据": ["stock_xg", "stock_ipo_", "stock_new_"],
    "千股千评": ["stock_comment_"],
    "盘口异动": ["stock_changes_"],
    "大宗交易": ["stock_dzjy_"],
    "停复牌": ["stock_tfp_"],
    "私募基金": ["amac_"],
    "奇货可查": ["qhkc_"],
}


def categorize(name):
    """Categorize a function by its prefix"""
    # Check specific categories first (longer prefixes)
    for cat, prefixes in CATEGORIES.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                return cat
    # Fallback
    if name.startswith("stock_"):
        return "股票数据"
    if name.startswith("fund_"):
        return "基金数据"
    if name.startswith("bond_"):
        return "债券数据"
    if name.startswith("macro_"):
        return "宏观数据"
    if name.startswith("index_"):
        return "指数数据"
    if name.startswith("futures_"):
        return "期货数据"
    return "其他"


def format_params(params):
    """Format params dict for display"""
    if not params:
        return "无需参数"
    parts = []
    for k, v in params.items():
        parts.append(f'{k}="{v}"')
    return ", ".join(parts)


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Separate success and others
    success = [r for r in results if r["status"] == "success"]
    timeout = [r for r in results if r["status"] == "timeout"]
    errors = [r for r in results if r["status"] not in ("success", "success_empty", "timeout", "skip_slow", "skip_params")]
    skipped = [r for r in results if r["status"] in ("skip_slow", "skip_params")]
    
    # Group successes by category
    grouped = defaultdict(list)
    for r in success:
        cat = categorize(r["name"])
        grouped[cat].append(r)
    
    # Also group all results for summary
    all_cats = defaultdict(lambda: {"success": 0, "error": 0, "timeout": 0, "skip": 0})
    for r in results:
        cat = categorize(r["name"])
        if r["status"] == "success":
            all_cats[cat]["success"] += 1
        elif r["status"] == "timeout":
            all_cats[cat]["timeout"] += 1
        elif r["status"] in ("skip_slow", "skip_params"):
            all_cats[cat]["skip"] += 1
        else:
            all_cats[cat]["error"] += 1
    
    # Write markdown
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(f"# AKShare 数据接口测试报告\n\n")
        f.write(f"> 生成时间: {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"> AKShare 版本: 1.18.11\n")
        f.write(f"> 测试总数: {len(results)} | ✅ 成功: {len(success)} | ⏱ 超时: {len(timeout)} | ❌ 失败: {len(errors)} | ⊘ 跳过: {len(skipped)}\n\n")
        
        # Table of contents
        f.write("## 目录\n\n")
        cat_order = sorted(grouped.keys(), key=lambda c: -len(grouped[c]))
        for cat in cat_order:
            count = len(grouped[cat])
            anchor = cat.replace(" ", "-").lower()
            f.write(f"- [{cat}](#{anchor}) ({count}个接口)\n")
        f.write(f"- [分类统计汇总](#分类统计汇总)\n")
        f.write(f"- [失败接口列表](#失败接口列表)\n\n")
        f.write("---\n\n")
        
        # Each category
        for cat in cat_order:
            items = grouped[cat]
            f.write(f"## {cat}\n\n")
            
            stats = all_cats[cat]
            f.write(f"✅ {stats['success']} 成功 | ❌ {stats['error']} 失败 | ⏱ {stats['timeout']} 超时 | ⊘ {stats['skip']} 跳过\n\n")
            
            f.write("| 接口名称 | 数据规模 | 示例参数 | 返回字段 |\n")
            f.write("|---------|---------|---------|--------|\n")
            
            for r in sorted(items, key=lambda x: x["name"]):
                name = r["name"]
                shape = r.get("data_shape", "")
                params = format_params(r.get("params", {}))
                cols = r.get("columns", [])
                cols_str = ", ".join(str(c) for c in cols[:8])
                if len(cols) > 8:
                    cols_str += f"... (+{len(cols)-8})"
                
                # Escape pipe chars in table
                params = params.replace("|", "\\|")
                cols_str = cols_str.replace("|", "\\|")
                
                f.write(f"| `{name}` | {shape} | {params} | {cols_str} |\n")
            
            f.write("\n")
        
        # Summary table
        f.write("## 分类统计汇总\n\n")
        f.write("| 分类 | 成功 | 失败 | 超时 | 跳过 | 成功率 |\n")
        f.write("|-----|------|------|------|------|-------|\n")
        for cat in sorted(all_cats.keys()):
            s = all_cats[cat]
            total = s["success"] + s["error"] + s["timeout"] + s["skip"]
            rate = f'{s["success"]/total*100:.0f}%' if total > 0 else "N/A"
            f.write(f"| {cat} | {s['success']} | {s['error']} | {s['timeout']} | {s['skip']} | {rate} |\n")
        f.write("\n")
        
        # Failed interfaces
        f.write("## 失败接口列表\n\n")
        f.write("### 超时接口\n\n")
        for r in sorted(timeout, key=lambda x: x["name"]):
            f.write(f"- `{r['name']}`\n")
        
        f.write("\n### 错误接口\n\n")
        f.write("| 接口 | 错误类型 | 错误信息 |\n")
        f.write("|-----|---------|--------|\n")
        for r in sorted(errors, key=lambda x: x["name"]):
            etype = r.get("error_type", r.get("status", ""))
            emsg = r.get("error_msg", r.get("error", ""))
            if emsg:
                emsg = emsg[:80].replace("|", "\\|").replace("\n", " ")
            f.write(f"| `{r['name']}` | {etype} | {emsg} |\n")
        
        f.write("\n### 跳过接口 (已知慢速/需登录)\n\n")
        for r in sorted(skipped, key=lambda x: x["name"]):
            f.write(f"- `{r['name']}` ({r['status']})\n")
        
        f.write("\n---\n")
        f.write(f"\n> ⚠️ 注意: 含 `_em` 后缀的接口来自东方财富，可能因反爬被封。\n")
        f.write(f"> 建议优先使用新浪 (`_sina`) 和同花顺 (`_ths`) 数据源。\n")
    
    print(f"Generated {OUTPUT}")
    print(f"Total: {len(results)}, Success: {len(success)}, Timeout: {len(timeout)}, Error: {len(errors)}, Skip: {len(skipped)}")


if __name__ == "__main__":
    main()
