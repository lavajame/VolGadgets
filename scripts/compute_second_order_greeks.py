import os
import math
import pandas as pd
import numpy as np
from bs_pricer import black_scholes_price, black_scholes_greeks


def compute_second_order(subfolder: str, date: str = None, eps_S_frac: float = 1e-3, eps_sigma: float = 1e-4, market_csv: str = None):
    base = os.path.join('backtests', subfolder)
    if market_csv:
        mkt_path = market_csv
    else:
        mkt_path = os.path.join(base, 'mkt_data.csv')
    if not os.path.exists(mkt_path):
        print('Missing market data:', mkt_path)
        return

    mkt = pd.read_csv(mkt_path, parse_dates=['quote_date', 'expiration'], dtype={'contract': str})
    # choose date: if provided use that, else pick first date with near_grid
    if date is None:
        dates = sorted(mkt['quote_date'].dropna().unique())
        start = None
        for d in dates:
            if ((mkt['quote_date'] == d) & (mkt['selection_group'] == 'near_grid')).any():
                start = pd.to_datetime(d)
                break
        if start is None:
            print('No start date with near_grid found')
            return
    else:
        start = pd.to_datetime(date)

    rows = []
    near0 = mkt[(mkt['quote_date'] == start) & (mkt['selection_group'] == 'near_grid')].copy()
    if near0.empty:
        print('No near_grid rows on date', start)
        return

    # also get far_target call contract for second-order greeks (use same start date filter)
    far_target = mkt[(mkt['quote_date'] == start) & (mkt['selection_group'] == 'far_target') & (mkt['option_type'] == 'call')].copy()
    if not far_target.empty:
        print(f'Found {len(far_target)} far_target call contract(s) at {start}')
    
    # combine near_grid and far_target contracts
    contracts_to_process = pd.concat([near0, far_target], ignore_index=True) if not far_target.empty else near0

    for _, r in contracts_to_process.iterrows():
        contract = r.get('contract', '?')
        try:
            S = float(r['underlying_close'])
            K = float(r['strike'])
            exp = pd.to_datetime(r['expiration'])
            T = max((exp - start).days / 365.0, 1/365.0)
            iv = float(r['implied_volatility']) if pd.notna(r['implied_volatility']) else 0.0
            opt = r['option_type']
        except Exception as e:
            print(f'  Skipping {contract}: parse error {e}')
            continue

        if T <= 0 or iv <= 0 or S <= 0:
            print(f'  Skipping {contract}: T={T}, iv={iv}, S={S}')
            continue

        # first-order greeks
        delta, gamma, vega, theta_year = black_scholes_greeks(S, K, T, 0.0, 0.0, iv, opt)

        # numeric epsilons
        eps_S = max(eps_S_frac * S, 0.01)
        eps_sig = eps_sigma

        # vanna = dVega/dS (or dDelta/dsigma) numeric
        # compute vega at S+eps and S-eps
        _, _, vega_p, _ = black_scholes_greeks(S + eps_S, K, T, 0.0, 0.0, iv, opt)
        _, _, vega_m, _ = black_scholes_greeks(S - eps_S, K, T, 0.0, 0.0, iv, opt)
        vanna = (vega_p - vega_m) / (2.0 * eps_S)

        # volga (vomma) = dVega/dsigma
        _, _, vega_sp, _ = black_scholes_greeks(S, K, T, 0.0, 0.0, iv + eps_sig, opt)
        _, _, vega_sm, _ = black_scholes_greeks(S, K, T, 0.0, 0.0, iv - eps_sig, opt)
        volga = (vega_sp - vega_sm) / (2.0 * eps_sig)

        # d theta / d spot (theta returned per-year) numeric
        _, _, _, theta_p = black_scholes_greeks(S + eps_S, K, T, 0.0, 0.0, iv, opt)
        _, _, _, theta_m = black_scholes_greeks(S - eps_S, K, T, 0.0, 0.0, iv, opt)
        dtheta_dS_year = (theta_p - theta_m) / (2.0 * eps_S)

        rows.append({
            'contract': r['contract'],
            'strike': K,
            'option_type': opt,
            'S': S,
            'T_years': T,
            'iv': iv,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta_year': theta_year,
            'vanna': vanna,
            'volga': volga,
            'dtheta_dS_year': dtheta_dS_year
        })

    out_df = pd.DataFrame(rows).set_index('contract')
    out_path = os.path.join(base, 'second_order_greeks_day0.csv')
    out_df.to_csv(out_path)
    
    print(f"\n{'='*110}")
    print(f"Second-Order Greeks at {start.date()}")
    print(f"{'='*110}\n")
    
    # Display full table with nice formatting
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(out_df.to_string())
    print(f"\n{'='*110}")
    print(f'Wrote: {out_path}')
    print(f"{'='*110}\n")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('subfolder')
    p.add_argument('--date', help='Date to compute greeks for (YYYY-MM-DD). Defaults to first date with near_grid')
    p.add_argument('--market-csv', help='Custom market data CSV (default: backtests/<subfolder>/mkt_data.csv)')
    args = p.parse_args()
    compute_second_order(args.subfolder, date=args.date, market_csv=args.market_csv)
