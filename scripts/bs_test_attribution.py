import pandas as pd
import numpy as np
from datetime import datetime
from bs_pricer import black_scholes_price


def days_to_years(d1, d2):
    return (d2 - d1).days / 365.0


def sequential_reprice(S0, S1, iv0, iv1, T0, T1, K, r, q, opt_type):
    # start price
    p0 = black_scholes_price(S0, K, T0, r, q, iv0, opt_type)
    # apply spot change (S1) keeping iv0, T0
    p_s = black_scholes_price(S1, K, T0, r, q, iv0, opt_type)
    # apply iv change to iv1 keeping S1, T0
    p_iv = black_scholes_price(S1, K, T0, r, q, iv1, opt_type)
    # apply time decay to T1 keeping S1, iv1
    p_t = black_scholes_price(S1, K, T1, r, q, iv1, opt_type)

    contrib_spot = p_s - p0
    contrib_iv = p_iv - p_s
    contrib_time = p_t - p_iv
    total = p_t - p0
    residual = total - (contrib_spot + contrib_iv + contrib_time)
    return contrib_spot, contrib_iv, contrib_time, residual, total


def main():
    base = 'backtests/130206_SPX'
    mkt = pd.read_csv(base + '/mkt_data.csv', parse_dates=['quote_date', 'expiration'], dtype={'contract': str})
    weights = pd.read_csv(base + '/near_weights.csv', dtype={'contract': str})

    contracts = list(weights['contract'])
    wmap = dict(zip(weights['contract'], weights['weight']))

    # pick first two dates with data
    dates = sorted(mkt['quote_date'].dropna().unique())
    if len(dates) < 2:
        print('Not enough dates')
        return
    d0 = dates[0]
    d1 = dates[1]

    m = mkt[(mkt['selection_group'] == 'near_grid') & (mkt['contract'].isin(contracts))]
    m0 = m[m['quote_date'] == d0].set_index('contract')
    m1 = m[m['quote_date'] == d1].set_index('contract')

    r = 0.0
    q = 0.0

    totals = {'spot': 0.0, 'iv': 0.0, 'time': 0.0, 'residual': 0.0, 'total': 0.0}

    for c in contracts:
        if c not in m0.index or c not in m1.index:
            continue
        row0 = m0.loc[c]
        row1 = m1.loc[c]
        S0 = float(row0['underlying_close'])
        S1 = float(row1['underlying_close'])
        iv0 = float(row0['implied_volatility']) if pd.notna(row0['implied_volatility']) else 0.0
        iv1 = float(row1['implied_volatility']) if pd.notna(row1['implied_volatility']) else 0.0
        K = float(row0['strike'])
        exp = pd.to_datetime(row0['expiration'])
        T0 = max(days_to_years(d0, exp), 1/365.0)
        T1 = max(days_to_years(d1, exp), 0.0)
        opt_type = row0['option_type']

        spot_c, iv_c, time_c, res_c, tot = sequential_reprice(S0, S1, iv0, iv1, T0, T1, K, r, q, opt_type)
        w = wmap.get(c, 0.0)
        # portfolio is short the near options
        totals['spot'] += - w * spot_c
        totals['iv'] += - w * iv_c
        totals['time'] += - w * time_c
        totals['residual'] += - w * res_c
        totals['total'] += - w * tot

    print('BS sequential attribution from d0->d1:')
    print(totals)


if __name__ == '__main__':
    main()
