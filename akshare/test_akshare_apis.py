"""
AKShare API 批量测试脚本
测试所有 akshare 公开接口，记录可用接口及其功能
"""
import akshare as ak
import pandas as pd
import json
import time
import signal
import inspect
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import wraps
import threading

# ============== 配置 ==============
TIMEOUT_SECONDS = 15  # 每个接口的超时时间
MAX_WORKERS = 1       # 串行测试，避免被封IP
OUTPUT_JSON = "akshare_test_results.json"

# ============== 常见测试参数映射 ==============
# 根据函数名模式提供测试参数
PARAM_HINTS = {
    # 股票相关 - 用平安银行000001 或 贵州茅台600519
    r"stock_zh_a_hist": {"symbol": "000001", "period": "daily", "start_date": "20250101", "end_date": "20250110", "adjust": ""},
    r"stock_zh_a_spot_em": {},
    r"stock_zh_a_spot": {},
    r"stock_zh_b_spot_em": {},
    r"stock_zh_a_daily": {"symbol": "sz000001", "start_date": "20250101", "end_date": "20250110", "adjust": "qfq"},
    r"stock_zh_a_minute": {"symbol": "sz000001", "period": "5"},
    r"stock_zh_a_tick_tx": {"symbol": "sz000001", "trade_date": "20250110"},
    r"stock_zh_index_daily": {"symbol": "sh000001"},
    r"stock_zh_index_spot_em": {},
    r"stock_zh_index_spot_sina": {"symbol": "大型指数"},
    r"stock_info_a_code_name": {},
    r"stock_info_sh_name_code": {"symbol": "主板A股"},
    r"stock_info_sz_name_code": {"symbol": "A股列表"},
    r"stock_info_bj_name_code": {},
    r"stock_board_concept_name_ths": {},
    r"stock_board_industry_name_ths": {},
    r"stock_board_concept_name_em": {},
    r"stock_board_industry_name_em": {},
    r"stock_zt_pool_em": {"date": "20250110"},
    r"stock_zt_pool_previous_em": {"date": "20250110"},
    r"stock_zt_pool_strong_em": {"date": "20250110"},
    r"stock_zt_pool_sub_new_em": {"date": "20250110"},
    r"stock_zt_pool_zbgc_em": {"date": "20250110"},
    r"stock_zt_pool_dtgc_em": {"date": "20250110"},
    r"stock_changes_em": {"symbol": "大笔买入"},
    r"stock_lhb_detail_em": {"start_date": "20250101", "end_date": "20250110"},
    r"stock_margin_sse": {"start_date": "20250101", "end_date": "20250110"},
    r"stock_margin_szse": {"start_date": "20250101", "end_date": "20250110"},
    r"stock_hsgt_north_net_flow_in_em": {"symbol": "北上"},
    # ETF/基金
    r"fund_etf_hist_sina": {"symbol": "sz513180"},
    r"fund_etf_spot_em": {},
    r"fund_lof_spot_em": {},
    r"fund_open_fund_daily_em": {},
    r"fund_name_em": {},
    r"fund_etf_fund_daily_em": {},
    r"fund_etf_hist_em": {"symbol": "513180", "period": "daily", "start_date": "20250101", "end_date": "20250110", "adjust": ""},
    # 期货
    r"futures_zh_spot": {},
    r"futures_main_sina": {"symbol": "V0"},
    r"futures_zh_daily_sina": {"symbol": "V0"},
    r"futures_zh_minute_sina": {"symbol": "V2501", "period": "5"},
    r"futures_display_name_list": {},
    r"futures_comm_info": {"symbol": "沪铜"},
    # 期权
    r"option_current_em": {},
    r"option_sse_list_sina": {"symbol": "50ETF", "exchange": "null"},
    r"option_finance_board": {"symbol": "嘉实沪深300ETF期权", "end_month": "2502"},
    # 债券
    r"bond_zh_hs_spot": {},
    r"bond_zh_hs_cov_spot": {},
    r"bond_cb_jsl": {},
    # 外汇
    r"fx_spot_quote": {},
    r"fx_pair_quote": {"symbol": "美元/人民币", "start_date": "20250101", "end_date": "20250110"},
    r"currency_latest": {"base": "USD", "symbols": "CNY"},
    # 利率
    r"rate_interbank": {"market": "上海银行同业拆借市场", "symbol": "Shibor人民币", "indicator": "1月"},
    # 宏观
    r"macro_china_gdp": {},
    r"macro_china_cpi": {},
    r"macro_china_ppi": {},
    r"macro_china_pmi": {},
    r"macro_china_money_supply": {},
    r"macro_china_lpr": {},
    r"macro_usa_gdp": {},
    r"macro_usa_cpi_monthly": {},
    r"macro_usa_unemployment_rate": {},
    r"macro_euro_gdp_yoy": {},
    r"macro_bank_usa_interest_rate": {},
    # 指数
    r"index_zh_a_hist": {"symbol": "000001", "period": "daily", "start_date": "20250101", "end_date": "20250110"},
    r"index_stock_info": {},
    r"index_stock_cons": {"symbol": "000300"},
    r"index_stock_cons_weight_csindex": {"symbol": "000300", "start_date": "20250101"},
    # 加密货币
    r"crypto_bitcoin_cme": {},
    # 申万
    r"sw_index_first_info": {},
    r"sw_index_second_info": {},
    r"sw_index_third_info": {},
    # 工具
    r"tool_trade_date_hist_sina": {},
    # 现货
    r"spot_golden_benchmark_sge": {},
    r"spot_silver_benchmark_sge": {},
    # 可转债
    r"bond_cov_comparison": {},
    r"bond_cb_redeem_jsl": {},
    # 年报季报
    r"stock_yjbb_em": {"date": "20240930"},
    r"stock_yysj_em": {"date": "20240930"},
    # 新股
    r"stock_xgsglb_em": {"symbol": "全部股票"},
    # 板块
    r"stock_board_concept_hist_em": {"symbol": "BK0493", "period": "daily", "start_date": "20250101", "end_date": "20250110", "adjust": ""},
    r"stock_board_industry_hist_em": {"symbol": "BK1027", "period": "daily", "start_date": "20250101", "end_date": "20250110", "adjust": ""},
    # 股东
    r"stock_gdfx_free_holding_statistics_em": {"date": "20240930"},
    r"stock_gdfx_holding_statistics_em": {"date": "20240930"},
    # 排污权
    r"index_ew_spot_price": {},
    # 空气
    r"air_quality_hist": {"city": "北京", "period": "day", "start_date": "20250101", "end_date": "20250110"},
}


def get_all_functions():
    """获取 akshare 所有公开可调用函数"""
    funcs = []
    for name in sorted(dir(ak)):
        if name.startswith('_'):
            continue
        obj = getattr(ak, name, None)
        if callable(obj) and not isinstance(obj, type):
            funcs.append(name)
    return funcs


def get_func_signature(func_name):
    """获取函数签名信息"""
    try:
        func = getattr(ak, func_name)
        sig = inspect.signature(func)
        params = {}
        for pname, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                params[pname] = {"required": True, "default": None}
            else:
                params[pname] = {"required": False, "default": str(param.default)}
        return params
    except Exception:
        return {}


def guess_params(func_name):
    """根据函数名猜测合理的测试参数"""
    # 1. 精确匹配
    if func_name in PARAM_HINTS:
        return PARAM_HINTS[func_name]
    
    # 2. 获取函数签名，对无参数的直接返回空
    try:
        func = getattr(ak, func_name)
        sig = inspect.signature(func)
        required_params = [
            p for p, v in sig.parameters.items()
            if v.default is inspect.Parameter.empty
        ]
        if not required_params:
            return {}
    except Exception:
        return {}
    
    # 3. 基于函数名模式和参数名猜测
    params = {}
    for p in required_params:
        if p == "symbol":
            if "stock" in func_name and ("us" in func_name or "美" in func_name):
                params[p] = "AAPL"
            elif "stock" in func_name and ("hk" in func_name or "港" in func_name):
                params[p] = "00700"
            elif "index" in func_name:
                params[p] = "000001"
            elif "futures" in func_name or "future" in func_name:
                params[p] = "V0"
            elif "fund" in func_name or "etf" in func_name:
                params[p] = "510300"
            elif "bond" in func_name:
                params[p] = "sz128038"
            elif "fx" in func_name or "currency" in func_name:
                params[p] = "美元/人民币"
            elif "option" in func_name:
                params[p] = "50ETF"
            elif "board" in func_name and "concept" in func_name:
                params[p] = "BK0493"
            elif "board" in func_name and "industry" in func_name:
                params[p] = "BK1027"
            elif "crypto" in func_name or "bitcoin" in func_name:
                params[p] = "btc"
            elif "spot" in func_name:
                params[p] = "Au99.99"
            else:
                params[p] = "000001"
        elif p == "date":
            params[p] = "20250110"
        elif p == "start_date":
            params[p] = "20250101"
        elif p == "end_date":
            params[p] = "20250110"
        elif p == "period":
            params[p] = "daily"
        elif p == "adjust":
            params[p] = ""
        elif p == "market":
            params[p] = "上海银行同业拆借市场"
        elif p == "indicator":
            params[p] = "今值"
        elif p == "year":
            params[p] = "2024"
        elif p == "quarter":
            params[p] = "4"
        elif p == "exchange":
            params[p] = "上交所"
        elif p == "timeout":
            params[p] = 10
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
        else:
            # 无法猜测的参数，跳过
            return None
    
    return params


def test_one_function(func_name):
    """测试单个函数"""
    result = {
        "name": func_name,
        "status": "unknown",
        "error": None,
        "data_shape": None,
        "params_used": None,
        "signature": get_func_signature(func_name),
        "time_taken": 0,
    }
    
    params = guess_params(func_name)
    if params is None:
        result["status"] = "skipped_no_params"
        result["error"] = "Cannot guess required parameters"
        return result
    
    result["params_used"] = params
    
    start = time.time()
    try:
        func = getattr(ak, func_name)
        data = func(**params)
        elapsed = time.time() - start
        result["time_taken"] = round(elapsed, 2)
        
        if isinstance(data, pd.DataFrame):
            result["status"] = "success"
            result["data_shape"] = f"{data.shape[0]} rows x {data.shape[1]} cols"
            result["columns"] = list(data.columns)[:20]
        elif isinstance(data, pd.Series):
            result["status"] = "success"
            result["data_shape"] = f"Series({len(data)})"
        elif isinstance(data, (list, dict)):
            result["status"] = "success"
            result["data_shape"] = f"{type(data).__name__}({len(data)})"
        elif data is not None:
            result["status"] = "success"
            result["data_shape"] = str(type(data).__name__)
        else:
            result["status"] = "success_empty"
            result["data_shape"] = "None"
    except TypeError as e:
        result["status"] = "param_error"
        result["error"] = str(e)[:200]
        result["time_taken"] = round(time.time() - start, 2)
    except KeyError as e:
        result["status"] = "key_error"
        result["error"] = str(e)[:200]
        result["time_taken"] = round(time.time() - start, 2)
    except ConnectionError as e:
        result["status"] = "connection_error"
        result["error"] = str(e)[:200]
        result["time_taken"] = round(time.time() - start, 2)
    except Exception as e:
        err_type = type(e).__name__
        result["status"] = f"error_{err_type}"
        result["error"] = f"{err_type}: {str(e)[:200]}"
        result["time_taken"] = round(time.time() - start, 2)
    
    return result


def test_with_timeout(func_name, timeout=TIMEOUT_SECONDS):
    """带超时的测试"""
    result_container = [None]
    
    def worker():
        result_container[0] = test_one_function(func_name)
    
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    
    if t.is_alive():
        return {
            "name": func_name,
            "status": "timeout",
            "error": f"Timed out after {timeout}s",
            "data_shape": None,
            "params_used": None,
            "signature": get_func_signature(func_name),
            "time_taken": timeout,
        }
    
    return result_container[0] if result_container[0] else {
        "name": func_name,
        "status": "unknown_error",
        "error": "No result returned",
        "data_shape": None,
        "params_used": None,
        "signature": {},
        "time_taken": 0,
    }


def main():
    all_funcs = get_all_functions()
    # 排除异常类
    all_funcs = [f for f in all_funcs if not any(f.endswith(x) for x in ['Error', 'Exception'])]
    
    print(f"[{datetime.now():%H:%M:%S}] 总共 {len(all_funcs)} 个函数待测试")
    print(f"超时设置: {TIMEOUT_SECONDS}s/函数")
    print("=" * 60)
    
    results = []
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, func_name in enumerate(all_funcs, 1):
        print(f"[{i}/{len(all_funcs)}] 测试 {func_name}...", end=" ", flush=True)
        
        result = test_with_timeout(func_name)
        results.append(result)
        
        if result["status"] == "success":
            success_count += 1
            print(f"✓ {result['data_shape']} ({result['time_taken']}s)")
        elif result["status"] in ("skipped_no_params", "param_error"):
            skip_count += 1
            print(f"⊘ {result.get('error', '')[:60]}")
        elif result["status"] == "timeout":
            fail_count += 1
            print(f"⏱ 超时")
        else:
            fail_count += 1
            print(f"✗ {result.get('error', '')[:60]}")
        
        # 每100个保存一次中间结果
        if i % 100 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n--- 中间保存: {success_count}成功 / {fail_count}失败 / {skip_count}跳过 ---\n")
        
        # 请求间隔，避免被封
        time.sleep(0.3)
    
    # 最终保存
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"测试完成! 成功: {success_count} / 失败: {fail_count} / 跳过: {skip_count}")
    print(f"总计: {len(results)} / {len(all_funcs)}")
    print(f"结果保存至: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
