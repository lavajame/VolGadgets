import os
import pandas as pd
import numpy as np
from bs_pricer import black_scholes_price, black_scholes_greeks


def check_moves(subfolder: str, date0: str, date1: str, delta_hedge: bool=False):
    base = os.path.join('backtests', subfolder)
    mkt_path = os.path.join(base, 'mkt_data.csv')
    weights_path = os.path.join(base, 'near_weights.csv')
    if not os.path.exists(mkt_path) or not os.path.exists(weights_path):
        print('Missing files')
        return

    mkt = pd.read_csv(mkt_path, parse_dates=['quote_date','expiration'], dtype={'contract': str})
    wdf = pd.read_csv(weights_path, dtype={'contract': str})
    contracts = list(wdf['contract'])
    wmap = dict(zip(wdf['contract'], wdf['weight']))

    d0 = pd.to_datetime(date0)
    d1 = pd.to_datetime(date1)

    rows = []
    totals = {'weighted_mid_pnl':0.0, 'weighted_bs_pnl':0.0, 'delta':0.0, 'gamma':0.0, 'vega':0.0, 'theta':0.0, 'residual':0.0}
    totals_pos = {'net_delta_pos': 0.0}

    for c in contracts:
        r0 = mkt[(mkt['quote_date']==d0) & (mkt['contract']==c) & (mkt['selection_group']=='near_grid')]
        r1 = mkt[(mkt['quote_date']==d1) & (mkt['contract']==c) & (mkt['selection_group']=='near_grid')]
        if r0.empty or r1.empty:
            continue
        r0 = r0.iloc[0]
        r1 = r1.iloc[0]

        S0 = float(r0['underlying_close'])
        S1 = float(r1['underlying_close'])
        K = float(r0['strike'])
        exp = pd.to_datetime(r0['expiration'])
        T0 = max((exp - d0).days/365.0, 1/365.0)
        T1 = max((exp - d1).days/365.0, 0.0)
        iv0 = float(r0['implied_volatility']) if pd.notna(r0['implied_volatility']) else 0.0
        iv1 = float(r1['implied_volatility']) if pd.notna(r1['implied_volatility']) else iv0
        mid0 = float(r0['mid'])
        mid1 = float(r1['mid'])
        w = float(wmap.get(c, 0.0))

        # BS repricing
        p0 = black_scholes_price(S0, K, T0, 0.0, 0.0, iv0, r0['option_type'])
        p1 = black_scholes_price(S1, K, T1, 0.0, 0.0, iv1, r1['option_type'])

        # greeks at day0
        delta0, gamma0, vega0, theta0_year = black_scholes_greeks(S0, K, T0, 0.0, 0.0, iv0, r0['option_type'])
        theta0 = theta0_year / 365.0

        dS = S1 - S0
        dIV = iv1 - iv0
        dt = (d1 - d0).days

        delta_pc = delta0 * dS
        gamma_pc = 0.5 * gamma0 * (dS**2)
        vega_pc = vega0 * dIV
        theta_pc = theta0 * dt

        bs_pnl = p1 - p0
        approx = delta_pc + gamma_pc + vega_pc + theta_pc
        residual = bs_pnl - approx

        # mid-based pnl (market mids)
        mid_pnl = mid1 - mid0

        # portfolio is short near: contribution = -w * pnl
        totals['weighted_mid_pnl'] += -w * mid_pnl
        totals['weighted_bs_pnl'] += -w * bs_pnl
        totals['delta'] += -w * delta_pc
        totals['gamma'] += -w * gamma_pc
        totals['vega'] += -w * vega_pc
        totals['theta'] += -w * theta_pc
        totals['residual'] += -w * residual
        # accumulate net delta position (short near: -w * delta0)
        totals_pos['net_delta_pos'] += -w * delta0

        rows.append({'contract': c, 'weight': w, 'mid0': mid0, 'mid1': mid1, 'mid_pnl': mid_pnl, 'iv0': iv0, 'iv1': iv1, 'dIV': dIV,
                     'p0': p0, 'p1': p1, 'bs_pnl': bs_pnl, 'approx': approx, 'residual': residual,
                     'delta_pc': delta_pc, 'gamma_pc': gamma_pc, 'vega_pc': vega_pc, 'theta_pc': theta_pc})

    df = pd.DataFrame(rows)
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    if df.empty:
        print('No matching contracts for dates')
        return

    print('\nPer-contract changes (near weights):')
    print(df[['contract','weight','mid0','mid1','mid_pnl','iv0','iv1','bs_pnl','approx','residual']].to_string(index=False))

    print('\nAggregated weighted contributions (near, short):')
    for k,v in totals.items():
        print(f'{k:20s}: {v:.6f}')

    if delta_hedge:
        # compute hedge pnl using underlying move (assume same underlying close across rows)
        try:
            S0_global = float(mkt[mkt['quote_date']==d0]['underlying_close'].iloc[0])
            S1_global = float(mkt[mkt['quote_date']==d1]['underlying_close'].iloc[0])
            dS_global = S1_global - S0_global
        except Exception:
            dS_global = 0.0

        hedge_size = -totals_pos['net_delta_pos']
        hedge_pnl = hedge_size * dS_global

        hedged = totals.copy()
        hedged['weighted_mid_pnl'] += hedge_pnl
        hedged['weighted_bs_pnl'] += hedge_pnl
        hedged['delta'] += hedge_pnl

        print('\nAggregated weighted contributions (near, short) WITH static day0 delta hedge:')
        print(f'net_delta_position (day0): {totals_pos["net_delta_pos"]:.6f}')
        print(f'hedge_size (units underlying): {hedge_size:.6f}, hedge_pnl: {hedge_pnl:.6f}')
        for k,v in hedged.items():
            print(f'{k:20s}: {v:.6f}')


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('subfolder')
    p.add_argument('--date0', default='2013-03-06')
    p.add_argument('--date1', default='2013-03-07')
    p.add_argument('--delta-hedge', action='store_true', help='Apply static day0 delta hedge to aggregated totals')
    args = p.parse_args()
    check_moves(args.subfolder, args.date0, args.date1, args.delta_hedge)
