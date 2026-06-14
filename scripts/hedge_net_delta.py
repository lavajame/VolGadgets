import os
import pandas as pd
import numpy as np


def hedge_net(subfolder: str, date0: str, date1: str, apply_hedge: bool=False):
    base = os.path.join('backtests', subfolder)
    evo_path = os.path.join(base, 'evolution.csv')
    if not os.path.exists(evo_path):
        print('Missing evolution.csv')
        return

    evo = pd.read_csv(evo_path, parse_dates=['quote_date'])
    d0 = pd.to_datetime(date0)
    d1 = pd.to_datetime(date1)
    r0 = evo[evo['quote_date'] == d0]
    r1 = evo[evo['quote_date'] == d1]
    if r0.empty or r1.empty:
        print('Dates not found in evolution.csv')
        return
    r0 = r0.iloc[0]
    r1 = r1.iloc[0]

    S0 = float(r0['underlying_close'])
    S1 = float(r1['underlying_close'])
    dS = S1 - S0
    dt = (d1 - d0).days

    net_mid0 = float(r0['net_mid'])
    net_mid1 = float(r1['net_mid'])
    net_mid_pnl = net_mid1 - net_mid0

    # net greeks at day0 (assume theta is per-year; convert to per-day)
    net_delta0 = float(r0.get('net_delta', 0.0))
    net_gamma0 = float(r0.get('net_gamma', 0.0))
    net_vega0 = float(r0.get('net_vega', 0.0))
    net_theta0 = float(r0.get('net_theta', 0.0))

    delta_pc = net_delta0 * dS
    gamma_pc = 0.5 * net_gamma0 * (dS ** 2)
    # for vega we don't have dIV at portfolio level here; approximate using change in near_eff_iv if present
    dIV = float(r1.get('near_eff_iv', 0.0)) - float(r0.get('near_eff_iv', 0.0))
    vega_pc = net_vega0 * dIV
    theta_pc = (net_theta0 / 365.0) * dt

    approx = delta_pc + gamma_pc + vega_pc + theta_pc

    print(f'Date pair: {date0} -> {date1}')
    print(f'Underlying S0={S0:.6f}, S1={S1:.6f}, dS={dS:.6f}, dt={dt} days')
    print(f'Net mid pnl (unhedged): {net_mid_pnl:.6f}')
    print(f'Net delta (day0): {net_delta0:.6f}')
    print(f'Approx contributions (delta,gamma,vega,theta): {delta_pc:.6f}, {gamma_pc:.6f}, {vega_pc:.6f}, {theta_pc:.6f}')
    print(f'Approx total (unhedged): {approx:.6f}')

    if apply_hedge:
        # static delta hedge placed at day0: hedge size = -net_delta0
        hedge_size = -net_delta0
        hedge_pnl = hedge_size * dS

        hedged_mid_pnl = net_mid_pnl + hedge_pnl
        hedged_approx = gamma_pc + vega_pc + theta_pc  # delta removed by hedge

        print(f'Hedge size (units underlying): {hedge_size:.6f}, hedge pnl: {hedge_pnl:.6f}')
        print(f'Net mid pnl (hedged): {hedged_mid_pnl:.6f}')
        print(f'Approx total (hedged, delta removed): {hedged_approx:.6f}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('subfolder')
    p.add_argument('--date0', default='2013-03-06')
    p.add_argument('--date1', default='2013-03-07')
    p.add_argument('--delta-hedge', action='store_true', help='Apply static day0 delta hedge')
    args = p.parse_args()
    hedge_net(args.subfolder, args.date0, args.date1, args.delta_hedge)
