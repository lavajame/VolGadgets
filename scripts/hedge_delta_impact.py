import os
import pandas as pd
import numpy as np
from plot_evolution import compute_pnl_factors


def compute_hedge_impact(subfolder: str):
    base = os.path.join('backtests', subfolder)
    weights_path = os.path.join(base, 'near_weights.csv')
    mkt_path = os.path.join(base, 'mkt_data.csv')
    if not os.path.exists(weights_path) or not os.path.exists(mkt_path):
        print('Missing files')
        return

    # unhedged delta contributions from compute_pnl_factors
    res = compute_pnl_factors(subfolder)
    if res is None:
        print('Could not compute pnl factors')
        return
    # compute_pnl_factors may return (pf, pf_cum) or (pf_near, pf_near_cum, pf_target, pf_target_cum)
    if len(res) == 2:
        pf_near, pf_near_cum = res
    else:
        pf_near, pf_near_cum, _, _ = res

    # read weights and market to compute portfolio delta at each close (using current date deltas)
    near = pd.read_csv(weights_path, dtype={'contract': str})
    mkt = pd.read_csv(mkt_path, parse_dates=['quote_date'], dtype={'contract': str})
    contracts = list(near['contract'].values)
    wmap = dict(zip(near['contract'].values, near['weight'].values))

    # build per-date near portfolio delta at close: -sum(w * delta)
    m = mkt[(mkt['selection_group'] == 'near_grid') & (mkt['contract'].isin(contracts))].copy()
    m['delta'] = pd.to_numeric(m['delta'], errors='coerce').fillna(0.0)
    dates = sorted(m['quote_date'].unique())
    near_delta_close = {}
    underlying = {}
    for d in dates:
        rows = m[m['quote_date'] == d]
        s = 0.0
        for _, r in rows.iterrows():
            s += wmap.get(r['contract'], 0.0) * r['delta']
        # portfolio delta (short) as used elsewhere is -sum(w * delta)
        near_delta_close[pd.to_datetime(d)] = -s
        # underlying close
        try:
            underlying[pd.to_datetime(d)] = float(rows['underlying_close'].iloc[0])
        except Exception:
            underlying[pd.to_datetime(d)] = np.nan

    # compute daily hedge pnl using daily rebalanced hedge: hedge_size_prev = -near_delta_close(prev_date)
    sorted_dates = sorted(near_delta_close.keys())
    hedge_pnl = []
    hedge_sizes = []
    prev_date = None
    hedge_size_prev = 0.0
    for d in sorted_dates:
        if prev_date is None:
            # on first day we set hedge at close; no pnl
            hedge_pnl.append(0.0)
            hedge_size_prev = -near_delta_close[d]
            hedge_sizes.append(hedge_size_prev)
            prev_date = d
            continue
        # pnl for period prev->d
        S_prev = underlying.get(prev_date, np.nan)
        S_cur = underlying.get(d, np.nan)
        dS = S_cur - S_prev if (not np.isnan(S_prev) and not np.isnan(S_cur)) else 0.0
        pnl = hedge_size_prev * dS
        hedge_pnl.append(pnl)
        # rebalance at close d
        hedge_size_prev = -near_delta_close[d]
        hedge_sizes.append(hedge_size_prev)
        prev_date = d

    hedge_pnl_series = pd.Series(hedge_pnl, index=sorted_dates)
    hedge_cum = hedge_pnl_series.cumsum()

    # unhedged cumulative delta pnl from pf_near_cum
    # pf_near index is quote_date
    unhedged_delta_cum = pf_near_cum['delta'] if 'delta' in pf_near_cum.columns else None

    # align dates: pf_near_cum may include same dates; convert index to datetime
    if unhedged_delta_cum is None:
        print('No delta column in computed pnl factors')
        return

    # build DataFrame combining
    df = pd.DataFrame({'unhedged_delta_cum': unhedged_delta_cum})
    df = df.reindex(sorted_dates).fillna(method='ffill').fillna(0.0)
    df['hedge_pnl_daily'] = hedge_pnl_series.reindex(sorted_dates).fillna(0.0)
    df['hedge_pnl_cum'] = df['hedge_pnl_daily'].cumsum()
    df['hedged_delta_cum'] = df['unhedged_delta_cum'] + df['hedge_pnl_cum']

    # print final cumulative totals
    last = df.iloc[-1]
    print(f"Unhedged cumulative delta PnL (short leg): {last['unhedged_delta_cum']:.6f}")
    print(f"Cumulative hedge PnL: {last['hedge_pnl_cum']:.6f}")
    print(f"Hedged cumulative delta PnL (short leg + hedge): {last['hedged_delta_cum']:.6f}")

    # save detailed CSV
    out = os.path.join(base, 'hedge_delta_impact.csv')
    df.to_csv(out, index_label='quote_date')
    print(f'Wrote detailed hedge impact to {out}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('subfolder')
    args = p.parse_args()
    compute_hedge_impact(args.subfolder)
