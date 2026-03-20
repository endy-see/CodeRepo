#!/usr/bin/env python3
"""
Finance Monster — 模块2: 中国衍生品
覆盖: 期货实时盘面、期货分钟K线+技术指标、期权IV/Greeks、期权RR25

数据源: 新浪财经
"""
import sys
import argparse

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    print("❌ 请先安装依赖: pip install akshare pandas")
    sys.exit(1)


# ═══════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_float(x):
    try:
        return float(str(x).replace('%', '').strip())
    except Exception:
        return None


def kv_df_to_dict(df, key_col='字段', val_col='值'):
    if df is None or len(df) == 0:
        return {}
    if key_col not in df.columns or val_col not in df.columns:
        key_col, val_col = df.columns[:2]
    out = {}
    for _, row in df.iterrows():
        k = str(row[key_col]).strip()
        v = row[val_col]
        if isinstance(v, str):
            v = v.strip()
        out[k] = v
    return out


def normalize_greeks_kv(kv):
    m = {
        '期权合约简称': 'name', '成交量': 'volume', '最新价': 'price_last',
        '行权价': 'strike', '隐含波动率': 'iv', 'Delta': 'delta',
        'Gamma': 'gamma', 'Theta': 'theta', 'Vega': 'vega',
        '理论价值': 'theo_value', '交易代码': 'trade_code',
    }
    out = {}
    for k, v in kv.items():
        if k in m:
            out[m[k]] = v
    for nk in ['volume', 'price_last', 'strike', 'iv', 'delta', 'gamma', 'theta', 'vega', 'theo_value']:
        if nk in out:
            out[nk] = to_float(out[nk])
    return out


# ═══════════════════════════════════════════════════
# 期货: 实时盘面
# ═══════════════════════════════════════════════════

def futures_board(symbol='PTA', top=10):
    df = ak.futures_zh_realtime(symbol=symbol)
    vol_cols = [c for c in df.columns if '成交' in c or c.lower() in ('volume',)]
    oi_cols = [c for c in df.columns if '持仓' in c or c.lower() in ('open_interest', 'oi')]
    sort_col = (vol_cols[0] if vol_cols else (oi_cols[0] if oi_cols else None))
    if sort_col:
        sdf = df.copy()
        sdf[sort_col] = pd.to_numeric(sdf[sort_col], errors='coerce')
        sdf = sdf.sort_values(sort_col, ascending=False).head(top)
    else:
        sdf = df.head(top)
    with pd.option_context('display.max_columns', 50, 'display.width', 200):
        print(f"📊 期货实时盘面 — {symbol} (共{len(df)}个合约, 显示前{top})\n")
        print(sdf.to_string(index=False))


# ═══════════════════════════════════════════════════
# 期货: 分钟K线 + 技术指标
# ═══════════════════════════════════════════════════

def futures_indicators(contract, period=5, tail=60):
    df = ak.futures_zh_minute_sina(symbol=contract, period=str(period))
    close_col = pick_col(df, ['close', '收盘', '收盘价', '最新价', 'Close'])
    if close_col is None:
        print(f"⚠️ 无法识别收盘价列: {list(df.columns)}")
        return
    close = pd.to_numeric(df[close_col], errors='coerce')

    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd_line
    df['Signal'] = signal
    df['Hist'] = macd_line - signal

    delta = close.diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta).clip(lower=0).rolling(14).mean()
    df['RSI14'] = 100 - (100 / (1 + rs))

    with pd.option_context('display.max_columns', 50, 'display.width', 200):
        print(f"📊 期货K线+指标 — {contract} ({period}分钟)\n")
        print(df.tail(tail).to_string(index=False))


# ═══════════════════════════════════════════════════
# 期权: IV + Greeks
# ═══════════════════════════════════════════════════

def options_greeks(underlying='510050', trade_date=None, n=10):
    if trade_date is None:
        sym = '50ETF' if underlying == '510050' else '300ETF'
        months = ak.option_sse_list_sina(symbol=sym)
        trade_date = months[0]

    call_df = ak.option_sse_codes_sina(symbol='看涨期权', trade_date=trade_date, underlying=underlying)
    put_df = ak.option_sse_codes_sina(symbol='看跌期权', trade_date=trade_date, underlying=underlying)

    codes = []
    if '期权代码' in call_df.columns:
        codes += call_df['期权代码'].astype(str).head(n // 2).tolist()
    if '期权代码' in put_df.columns:
        codes += put_df['期权代码'].astype(str).head(n - len(codes)).tolist()

    rows = []
    for code in codes:
        gdf = ak.option_sse_greeks_sina(symbol=code)
        kv = kv_df_to_dict(gdf)
        norm = normalize_greeks_kv(kv)
        norm['sina_code'] = code
        rows.append(norm)

    out = pd.DataFrame(rows)
    cols = [c for c in ['sina_code', 'name', 'trade_code', 'strike', 'price_last',
                         'iv', 'delta', 'gamma', 'theta', 'vega', 'volume', 'theo_value']
            if c in out.columns]

    with pd.option_context('display.max_columns', 50, 'display.width', 200):
        print(f"📊 期权IV+Greeks — underlying={underlying} trade_date={trade_date}\n")
        print(out[cols].to_string(index=False))


# ═══════════════════════════════════════════════════
# 期权: RR25 风险逆转
# ═══════════════════════════════════════════════════

def _fetch_side(trade_date, underlying, side):
    df = ak.option_sse_codes_sina(symbol=side, trade_date=trade_date, underlying=underlying)
    if '期权代码' not in df.columns:
        raise ValueError(f"列名异常: {list(df.columns)}")
    codes = df['期权代码'].astype(str).tolist()
    rows = []
    for code in codes:
        gdf = ak.option_sse_greeks_sina(symbol=code)
        kv = kv_df_to_dict(gdf)
        norm = normalize_greeks_kv(kv)
        norm['sina_code'] = code
        rows.append(norm)
    return pd.DataFrame(rows)


def options_rr25(underlying='510050', trade_date=None):
    if trade_date is None:
        sym = '50ETF' if underlying == '510050' else '300ETF'
        trade_date = ak.option_sse_list_sina(symbol=sym)[0]

    call = _fetch_side(trade_date, underlying, '看涨期权').dropna(subset=['delta', 'iv'])
    put = _fetch_side(trade_date, underlying, '看跌期权').dropna(subset=['delta', 'iv'])

    call['dist'] = (call['delta'] - 0.25).abs()
    put['dist'] = (put['delta'] + 0.25).abs()

    c = call.sort_values('dist').head(1).iloc[0].to_dict() if len(call) else None
    p = put.sort_values('dist').head(1).iloc[0].to_dict() if len(put) else None

    if not c or not p:
        print("⚠️ 数据不足，无法计算 RR25")
        return

    rr25 = c['iv'] - p['iv']
    print(f"📊 RR25 — underlying={underlying} trade_date={trade_date}")
    print(f"RR25 = iv_call25({c['iv']}) - iv_put25({p['iv']}) = {rr25}")
    print('\nCALL (+0.25Δ):')
    for k in ['sina_code', 'trade_code', 'name', 'strike', 'price_last', 'delta', 'iv', 'volume']:
        if k in c:
            print(f"  {k}: {c[k]}")
    print('\nPUT (-0.25Δ):')
    for k in ['sina_code', 'trade_code', 'name', 'strike', 'price_last', 'delta', 'iv', 'volume']:
        if k in p:
            print(f"  {k}: {p[k]}")


# ═══════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Finance Monster — 中国衍生品模块')
    sub = parser.add_subparsers(dest='command')

    # 期货盘面
    fb = sub.add_parser('futures-board', help='期货实时盘面')
    fb.add_argument('--symbol', default='PTA', help='品种名称 (PTA, IF, RB...)')
    fb.add_argument('--top', type=int, default=10)

    # 期货指标
    fi = sub.add_parser('futures-indicators', help='期货K线+技术指标')
    fi.add_argument('--contract', required=True, help='合约代码 (IF2603, RB2410...)')
    fi.add_argument('--period', type=int, default=5, choices=[1, 5, 15, 30, 60])
    fi.add_argument('--tail', type=int, default=60)

    # 期权Greeks
    og = sub.add_parser('options-greeks', help='期权IV+Greeks')
    og.add_argument('--underlying', default='510050', help='510050(50ETF) / 510300(300ETF)')
    og.add_argument('--trade-date', default=None, help='YYYYMM')
    og.add_argument('--n', type=int, default=10)

    # 期权RR25
    rr = sub.add_parser('options-rr25', help='期权RR25风险逆转')
    rr.add_argument('--underlying', default='510050')
    rr.add_argument('--trade-date', default=None)

    args = parser.parse_args()

    if args.command == 'futures-board':
        futures_board(args.symbol, args.top)
    elif args.command == 'futures-indicators':
        futures_indicators(args.contract, args.period, args.tail)
    elif args.command == 'options-greeks':
        options_greeks(args.underlying, args.trade_date, args.n)
    elif args.command == 'options-rr25':
        options_rr25(args.underlying, args.trade_date)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
