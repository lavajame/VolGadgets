#!/usr/bin/env python
"""
Vega Isolation Diagnostic
Measures how well the portfolio structure isolates vega exposure.
Compares multiple constraint configurations.
"""

import pandas as pd
import numpy as np
import sys
import os

def analyze_vega_isolation(subfolder):
    """
    Analyze vega isolation for a backtest.
    Returns metrics: delta_exposure, gamma_exposure, theta_exposure, vega_ratio, etc.
    """
    
    evo_path = os.path.join('backtests', subfolder, 'evolution.csv')
    if not os.path.exists(evo_path):
        print(f"ERROR: {evo_path} not found")
        return None
    
    evo = pd.read_csv(evo_path)
    
    # Day 0 Greeks (at-the-money exposure)
    row0 = evo[evo['quote_date'] == evo['quote_date'].min()].iloc[0]
    row1 = evo[evo['quote_date'] == evo['quote_date'].max()].iloc[0]
    
    net_delta_d0 = float(row0['net_delta']) if pd.notna(row0['net_delta']) else 0
    net_gamma_d0 = float(row0['net_gamma']) if pd.notna(row0['net_gamma']) else 0
    net_theta_d0 = float(row0['net_theta']) if pd.notna(row0['net_theta']) else 0
    net_vega_d0 = float(row0['net_vega']) if pd.notna(row0['net_vega']) else 0
    
    net_delta_d1 = float(row1['net_delta']) if pd.notna(row1['net_delta']) else 0
    net_gamma_d1 = float(row1['net_gamma']) if pd.notna(row1['net_gamma']) else 0
    net_theta_d1 = float(row1['net_theta']) if pd.notna(row1['net_theta']) else 0
    net_vega_d1 = float(row1['net_vega']) if pd.notna(row1['net_vega']) else 0
    
    # Actual PnL
    total_pnl = float(row1['net_mid'] - row0['net_mid'])
    
    # Spot and time movement
    S0 = float(row0['underlying_close'])
    S1 = float(row1['underlying_close'])
    dS = S1 - S0
    
    # Time decay: estimate from first and last date
    dates = pd.to_datetime(evo['quote_date']).unique()
    days_elapsed = (dates[-1] - dates[0]).days
    
    # Expected PnL from Greeks (using day 0 sensitivities)
    delta_pnl = net_delta_d0 * dS
    gamma_pnl = 0.5 * net_gamma_d0 * dS**2
    theta_pnl = net_theta_d0 * days_elapsed / 365.0
    
    # For vega: need IV change
    # We can estimate from evolution by looking at target vega pnl vs near vega pnl
    target_vega_d0 = float(row0['target_vega']) if pd.notna(row0['target_vega']) else 0
    near_vega_d0 = float(row0['near_vega_short']) if pd.notna(row0['near_vega_short']) else 0
    target_mid_d0 = float(row0['target_mid']) if pd.notna(row0['target_mid']) else 0
    target_mid_d1 = float(row1['target_mid']) if pd.notna(row1['target_mid']) else 0
    near_mid_d0 = float(row0['near_portfolio_mid_short']) if pd.notna(row0['near_portfolio_mid_short']) else 0
    near_mid_d1 = float(row1['near_portfolio_mid_short']) if pd.notna(row1['near_portfolio_mid_short']) else 0
    
    target_pnl_actual = target_mid_d1 - target_mid_d0
    near_pnl_actual = near_mid_d1 - near_mid_d0
    
    # Greeks-explained PnL
    greeks_explained = delta_pnl + gamma_pnl + theta_pnl
    residual = total_pnl - greeks_explained
    
    # If residual is largely vega-driven, then vega_pnl ≈ residual
    # The unexplained after delta/gamma/theta should be vega + cross-terms
    
    return {
        'subfolder': subfolder,
        'days': days_elapsed,
        'spot_move': dS,
        'total_pnl': total_pnl,
        'net_delta_d0': net_delta_d0,
        'net_gamma_d0': net_gamma_d0,
        'net_theta_d0': net_theta_d0,
        'net_vega_d0': net_vega_d0,
        'delta_pnl_est': delta_pnl,
        'gamma_pnl_est': gamma_pnl,
        'theta_pnl_est': theta_pnl,
        'greeks_explained_pnl': greeks_explained,
        'residual_pnl': residual,
        'residual_pct_of_total': 100.0 * residual / total_pnl if total_pnl != 0 else 0,
        'target_pnl': target_pnl_actual,
        'near_pnl': near_pnl_actual
    }


def print_comparison(results_list):
    """Print a comparison table of multiple configurations."""
    
    if not results_list:
        print("No results to compare")
        return
    
    print("\n" + "="*100)
    print("VEGA ISOLATION DIAGNOSTIC - CONFIGURATION COMPARISON")
    print("="*100)
    
    # Print headers
    print(f"\n{'Config':<35} {'Delta_d0':>10} {'Gamma_d0':>10} {'Theta_d0':>10} {'Vega_d0':>10}")
    print("-" * 100)
    
    for r in results_list:
        config_name = r.get('name', r['subfolder'])
        print(f"{config_name:<35} {r['net_delta_d0']:>10.6f} {r['net_gamma_d0']:>10.8f} {r['net_theta_d0']:>10.2f} {r['net_vega_d0']:>10.2f}")
    
    # Print PnL attribution
    print(f"\n{'Config':<35} {'Total_PnL':>12} {'Delta_PnL':>12} {'Gamma_PnL':>12} {'Theta_PnL':>12} {'Residual':>12} {'Residual%':>10}")
    print("-" * 100)
    
    for r in results_list:
        config_name = r.get('name', r['subfolder'])
        print(f"{config_name:<35} {r['total_pnl']:>12.2f} {r['delta_pnl_est']:>12.2f} {r['gamma_pnl_est']:>12.2f} {r['theta_pnl_est']:>12.2f} {r['residual_pnl']:>12.2f} {r['residual_pct_of_total']:>9.1f}%")
    
    # Print quality metrics
    print(f"\n{'Config':<35} {'DeltaExp@d0':>12} {'GammaExp@d0':>12} {'ThetaExp@d0':>12} {'VegaExp@d0':>12}")
    print("-" * 100)
    
    for r in results_list:
        config_name = r.get('name', r['subfolder'])
        delta_score = abs(r['net_delta_d0'])
        gamma_score = abs(r['net_gamma_d0'])
        theta_score = abs(r['net_theta_d0'])
        vega_score = abs(r['net_vega_d0'])
        print(f"{config_name:<35} {delta_score:>12.6f} {gamma_score:>12.8f} {theta_score:>12.2f} {vega_score:>12.2f}")
    
    print("\n" + "="*100)
    print("INTERPRETATION:")
    print("- Delta/Gamma/Theta@d0 should be SMALL (close to 0) for good isolation")
    print("- Residual PnL should be close to 100% (all PnL from vega, not unexplained)")
    print("- If residual% < 80%, there's unaccounted PnL (bid-ask spread, higher-order greeks, etc)")
    print("="*100)


def main():
    if len(sys.argv) < 2:
        print("Usage: python vega_isolation_diagnostic.py <subfolder1> [<subfolder2> ...] [--names name1 name2 ...]")
        print("Example: python vega_isolation_diagnostic.py 130301_SPX_Wide")
        print("  or: python vega_isolation_diagnostic.py 130301_SPX_Wide 130301_SPX_Wide --names 'delta;gamma' 'delta;gamma;theta;S_bounds'")
        sys.exit(1)
    
    subfolders = sys.argv[1:]
    names = []
    
    # Check for --names flag
    if '--names' in subfolders:
        idx = subfolders.index('--names')
        names = subfolders[idx+1:]
        subfolders = subfolders[:idx]
    
    # Pad names if not provided
    while len(names) < len(subfolders):
        names.append(f"Config {len(names)+1}")
    
    results_list = []
    for sf, name in zip(subfolders, names):
        print(f"\nAnalyzing {sf}...")
        r = analyze_vega_isolation(sf)
        if r:
            r['name'] = name
            results_list.append(r)
            
            # Print individual summary
            print(f"\n--- {name} ({sf}) ---")
            print(f"Days elapsed: {r['days']}")
            print(f"Spot move: {r['spot_move']:.2f}")
            print(f"\nDay-0 Greek exposures:")
            print(f"  Delta:  {r['net_delta_d0']:>10.6f}")
            print(f"  Gamma:  {r['net_gamma_d0']:>10.8f}")
            print(f"  Theta:  {r['net_theta_d0']:>10.2f} (per year)")
            print(f"  Vega:   {r['net_vega_d0']:>10.2f}")
            print(f"\nPnL attribution:")
            print(f"  Total PnL:      {r['total_pnl']:>10.2f}")
            print(f"  Delta contrib:  {r['delta_pnl_est']:>10.2f}")
            print(f"  Gamma contrib:  {r['gamma_pnl_est']:>10.2f}")
            print(f"  Theta contrib:  {r['theta_pnl_est']:>10.2f}")
            print(f"  Greeks total:   {r['greeks_explained_pnl']:>10.2f}")
            print(f"  Residual (vega+):  {r['residual_pnl']:>10.2f}  ({r['residual_pct_of_total']:.1f}% of total)")
            print(f"\nQuality score: {'✓ GOOD' if abs(r['net_delta_d0']) < 0.01 and abs(r['net_gamma_d0']) < 0.001 else '✗ NEEDS WORK'}")
    
    # Comparison table
    if len(results_list) > 1:
        print_comparison(results_list)
    
    return results_list


if __name__ == '__main__':
    main()
