"""
AKShare API 批量测试 - v2 (进程级超时，跳过慢接口)
"""
import akshare as ak
import pandas as pd
import json
import time
import inspect
import multiprocessing as mp
from datetime import datetime

TIMEOUT = 10
OUTPUT = "akshare_test_results.json"

# 已知超慢的函数前缀（内部分页几百次）
SLOW_PREFIXES = [
    "amac_",          # 基金业协会 - 分页太多
    "qhkc_",          # 奇货可查 - 需要登录
]

# 已知需要特殊环境的
SKIP_FUNCS = set([
    "stock_zh_a_tick_tx",       # 需要具体交易日期+时间
    "stock_zh_a_tick_163",      # 同上
])

PARAM_HINTS = {
    "air_city_table": {},
    "air_quality_hebei": {},
    "air_quality_hist": {"city": "北京", "period": "day", "start_date": "20250101", "end_date": "20250110"},
    "air_quality_rank": {},
    "air_quality_watch_point": {"city": "北京"},
    "stock_zh_a_spot_em": {},
    "stock_zh_b_spot_em": {},
    "stock_zh_a_hist": {"symbol": "000001", "period": "daily", "start_date": "20250101", "end_date": "20250110", "adjust": ""},
    "stock_zh_a_daily": {"symbol": "sz000001", "start_date": "20250101", "end_date": "20250110", "adjust": "qfq"},
    "stock_zh_a_minute": {"symbol": "sz000001", "period": "5"},
    "stock_zh_index_daily": {"symbol": "sh000001"},
    "stock_zh_index_spot_em": {},
    "stock_zh_index_spot_sina": {"symbol": "大型指数"},
    "stock_info_a_code_name": {},
    "stock_info_sh_name_code": {"symbol": "主板A股"},
    "stock_info_sz_name_code": {"symbol": "A股列表"},
    "stock_info_bj_name_code": {},
    "stock_board_concept_name_ths": {},
    "stock_board_industry_name_ths": {},
    "stock_board_concept_name_em": {},
    "stock_board_industry_name_em": {},
    "stock_zt_pool_em": {"date": "20250110"},
    "stock_zt_pool_previous_em": {"date": "20250110"},
    "stock_zt_pool_strong_em": {"date": "20250110"},
    "stock_zt_pool_sub_new_em": {"date": "20250110"},
    "stock_zt_pool_zbgc_em": {"date": "20250110"},
    "stock_zt_pool_dtgc_em": {"date": "20250110"},
    "stock_changes_em": {"symbol": "大笔买入"},
    "stock_lhb_detail_em": {"start_date": "20250101", "end_date": "20250110"},
    "stock_margin_sse": {"start_date": "20250101", "end_date": "20250110"},
    "stock_margin_szse": {"start_date": "20250101", "end_date": "20250110"},
    "stock_hsgt_north_net_flow_in_em": {"symbol": "北上"},
    "fund_etf_hist_sina": {"symbol": "sz513180"},
    "fund_etf_spot_em": {},
    "fund_lof_spot_em": {},
    "fund_open_fund_daily_em": {},
    "fund_name_em": {},
    "fund_etf_hist_em": {"symbol": "513180", "period": "daily", "start_date": "20250101", "end_date": "20250110", "adjust": ""},
    "futures_zh_spot": {},
    "futures_main_sina": {"symbol": "V0"},
    "futures_zh_daily_sina": {"symbol": "V0"},
    "futures_zh_minute_sina": {"symbol": "V2501", "period": "5"},
    "futures_display_name_list": {},
    "futures_comm_info": {"symbol": "沪铜"},
    "option_current_em": {},
    "bond_zh_hs_spot": {},
    "bond_zh_hs_cov_spot": {},
    "bond_cb_jsl": {},
    "fx_spot_quote": {},
    "currency_latest": {"base": "USD", "symbols": "CNY"},
    "macro_china_gdp": {},
    "macro_china_cpi": {},
    "macro_china_ppi": {},
    "macro_china_pmi": {},
    "macro_china_money_supply": {},
    "macro_china_lpr": {},
    "macro_usa_gdp": {},
    "macro_usa_cpi_monthly": {},
    "macro_usa_unemployment_rate": {},
    "macro_euro_gdp_yoy": {},
    "index_zh_a_hist": {"symbol": "000001", "period": "daily", "start_date": "20250101", "end_date": "20250110"},
    "index_stock_info": {},
    "index_stock_cons": {"symbol": "000300"},
    "crypto_bitcoin_cme": {},
    "sw_index_first_info": {},
    "sw_index_second_info": {},
    "sw_index_third_info": {},
    "tool_trade_date_hist_sina": {},
    "spot_golden_benchmark_sge": {},
    "spot_silver_benchmark_sge": {},
    "bond_cov_comparison": {},
    "bond_cb_redeem_jsl": {},
    "stock_yjbb_em": {"date": "20240930"},
    "stock_yysj_em": {"date": "20240930"},
    "stock_xgsglb_em": {"symbol": "全部股票"},
}


def guess_params(func_name):
    """根据函数名猜测测试参数"""
    if func_name in PARAM_HINTS:
        return PARAM_HINTS[func_name]
    try:
        func = getattr(ak, func_name)
        sig = inspect.signature(func)
        required = [p for p, v in sig.parameters.items() if v.default is inspect.Parameter.empty]
        if not required:
            return {}
    except:
        return {}
    
    params = {}
    for p in required:
        if p == "symbol":
            if "us_" in func_name or "_us" in func_name:
                params[p] = "AAPL"
            elif "_hk" in func_name or "hk_" in func_name:
                params[p] = "00700"
            elif "index" in func_name:
                params[p] = "000001"
            elif "futures" in func_name:
                params[p] = "V0"
            elif "fund" in func_name or "etf" in func_name:
                params[p] = "510300"
            elif "bond" in func_name:
                params[p] = "sz128038"
            elif "option" in func_name:
                params[p] = "50ETF"
            elif "board" in func_name and "concept" in func_name:
                params[p] = "BK0493"
            elif "board" in func_name and "industry" in func_name:
                params[p] = "BK1027"
            elif "fx" in func_name or "currency" in func_name:
                params[p] = "美元/人民币"
            elif "crypto" in func_name:
                params[p] = "btc"
            elif "spot" in func_name:
                params[p] = "Au99.99"
            elif "macro" in func_name:
                params[p] = "中国"
            elif "sw_index" in func_name:
                params[p] = "801010"
            elif "news" in func_name or "stock_news" in func_name:
                params[p] = "000001"
            else:
                params[p] = "000001"
        elif p == "date":
            params[p] = "20250110"
        elif p == "start_date":
            params[p] = "20250101"
        elif p == "end_date":
            params[p] = "20250310"
        elif p == "period":
            params[p] = "daily"
        elif p == "adjust":
            params[p] = ""
        elif p == "market":
            if "interbank" in func_name:
                params[p] = "上海银行同业拆借市场"
            else:
                params[p] = "上交所"
        elif p == "indicator":
            params[p] = "今值"
        elif p == "year":
            params[p] = "2024"
        elif p == "quarter":
            params[p] = "4"
        elif p == "exchange":
            params[p] = "上交所"
        elif p == "city":
            params[p] = "北京"
        elif p == "province":
            params[p] = "北京"
        elif p == "base":
            params[p] = "USD"
        elif p == "symbols":
            params[p] = "CNY"
        elif p == "code":
            params[p] = "000001"
        elif p == "name":
            params[p] = "平安银行"
        elif p == "stock":
            params[p] = "000001"
        elif p == "end_month":
            params[p] = "2503"
        elif p == "timeout":
            params[p] = 10
        elif p in ("page", "page_size"):
            params[p] = 1
        else:
            return None  # 无法猜测
    return params


def _worker(func_name, params, result_queue):
    """子进程内执行的函数"""
    import akshare as ak
    import pandas as pd
    import time
    
    start = time.time()
    try:
        func = getattr(ak, func_name)
        data = func(**params)
        elapsed = round(time.time() - start, 2)
        
        if isinstance(data, pd.DataFrame):
            result_queue.put({
                "name": func_name, "status": "success",
                "data_shape": f"{data.shape[0]}x{data.shape[1]}",
                "columns": list(data.columns)[:15],
                "params": params, "time": elapsed
            })
        elif isinstance(data, (pd.Series, list, dict)):
            n = len(data) if hasattr(data, '__len__') else 0
            result_queue.put({
                "name": func_name, "status": "success",
                "data_shape": f"{type(data).__name__}({n})",
                "columns": [], "params": params, "time": elapsed
            })
        elif data is not None:
            result_queue.put({
                "name": func_name, "status": "success",
                "data_shape": str(type(data).__name__),
                "columns": [], "params": params, "time": elapsed
            })
        else:
            result_queue.put({
                "name": func_name, "status": "success_empty",
                "data_shape": "None", "columns": [],
                "params": params, "time": elapsed
            })
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        result_queue.put({
            "name": func_name,
            "status": f"error",
            "error_type": type(e).__name__,
            "error_msg": str(e)[:150],
            "params": params, "time": elapsed,
            "data_shape": None, "columns": []
        })


def test_func(func_name):
    """带进程级超时的测试"""
    params = guess_params(func_name)
    if params is None:
        return {"name": func_name, "status": "skip_params", "data_shape": None, "columns": [], "params": None, "time": 0}
    
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(func_name, params, q), daemon=True)
    p.start()
    p.join(timeout=TIMEOUT)
    
    if p.is_alive():
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
        return {"name": func_name, "status": "timeout", "data_shape": None, "columns": [], "params": params, "time": TIMEOUT}
    
    if not q.empty():
        return q.get()
    return {"name": func_name, "status": "process_error", "data_shape": None, "columns": [], "params": params, "time": 0}


def get_all_functions():
    funcs = []
    for name in sorted(dir(ak)):
        if name.startswith('_'):
            continue
        obj = getattr(ak, name, None)
        if callable(obj) and not isinstance(obj, type):
            funcs.append(name)
    return funcs


def should_skip(name):
    if name in SKIP_FUNCS:
        return True
    for prefix in SLOW_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


if __name__ == "__main__":
    mp.freeze_support()
    
    all_funcs = get_all_functions()
    test_funcs = [f for f in all_funcs if not should_skip(f)]
    skipped = [f for f in all_funcs if should_skip(f)]
    
    print(f"[{datetime.now():%H:%M:%S}] 总函数: {len(all_funcs)}, 测试: {len(test_funcs)}, 跳过: {len(skipped)}")
    print(f"超时: {TIMEOUT}s/函数")
    print("=" * 70)
    
    results = []
    stats = {"success": 0, "error": 0, "timeout": 0, "skip": 0}
    
    for i, fn in enumerate(test_funcs, 1):
        print(f"[{i}/{len(test_funcs)}] {fn}...", end=" ", flush=True)
        
        r = test_func(fn)
        results.append(r)
        
        s = r["status"]
        if s == "success":
            stats["success"] += 1
            print(f"✓ {r['data_shape']} ({r['time']}s)")
        elif s == "timeout":
            stats["timeout"] += 1
            print("⏱ timeout")
        elif s.startswith("skip"):
            stats["skip"] += 1
            print("⊘ skip")
        else:
            stats["error"] += 1
            print(f"✗ {r.get('error_type','')} {r.get('error_msg','')[:50]}")
        
        # 保存中间结果
        if i % 50 == 0:
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n--- 进度 {i}/{len(test_funcs)}: ✓{stats['success']} ✗{stats['error']} ⏱{stats['timeout']} ⊘{stats['skip']} ---\n")
        
        time.sleep(0.2)
    
    # 添加跳过的函数
    for fn in skipped:
        results.append({"name": fn, "status": "skip_slow", "data_shape": None, "columns": [], "params": None, "time": 0})
    
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print(f"完成! ✓{stats['success']} ✗{stats['error']} ⏱{stats['timeout']} ⊘{stats['skip']}")
    print(f"结果: {OUTPUT}")
