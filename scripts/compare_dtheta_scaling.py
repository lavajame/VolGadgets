#!/usr/bin/env python
"""
Compare dtheta/dS scaling before and after fix.
"""

import numpy as np
from bs_pricer import black_scholes_greeks


def compare_scaling():
    """Compare unscaled vs scaled dtheta/dS."""
    S = 1462.42
    K = 1490.0
    T = 0.157534  # ~57 days
    r = 0.0
    q = 0.0
    sigma = 0.0288
    option_type = 'call'
    
    # Compute dtheta/dS via finite differences (per-year)
    eps_S_frac = 1e-3
    eps_S = eps_S_frac * S
    
    theta_up = black_scholes_greeks(S + eps_S, K, T, r, q, sigma, option_type)[3]
    theta_dn = black_scholes_greeks(S - eps_S, K, T, r, q, sigma, option_type)[3]
    dtheta_dS_year = (theta_up - theta_dn) / (2 * eps_S)
    
    print("dtheta/dS Scaling Fix Impact")
    print("=" * 70)
    print()
    
    # Simulate a 5 dollar move
    dS = 5.0
    dt = 1  # 1 day
    
    print(f"Scenario: S move of {dS}$, over {dt} day(s)")
    print(f"dtheta/dS (per-year): {dtheta_dS_year:.8f}")
    print()
    
    print("BEFORE FIX (incorrect scaling):")
    dtheta_dS_pc_wrong = dtheta_dS_year * dS
    print(f"  dtheta_dS_pc = dtheta_dS_year × dS")
    print(f"  dtheta_dS_pc = {dtheta_dS_year:.8f} × {dS}")
    print(f"  dtheta_dS_pc = {dtheta_dS_pc_wrong:.8f}  (units: $/year, too large!)")
    print(f"  This ~{dtheta_dS_pc_wrong:.2f}x overestimates daily impact by 365x factor")
    print()
    
    print("AFTER FIX (correct scaling):")
    dtheta_dS_daily = dtheta_dS_year / 365.0
    dtheta_dS_pc_correct = dtheta_dS_daily * dS
    print(f"  dtheta_dS_daily = dtheta_dS_year / 365")
    print(f"  dtheta_dS_daily = {dtheta_dS_year:.8f} / 365")
    print(f"  dtheta_dS_daily = {dtheta_dS_daily:.10f}  (units: $/day)")
    print(f"  dtheta_dS_pc = dtheta_dS_daily × dS")
    print(f"  dtheta_dS_pc = {dtheta_dS_daily:.10f} × {dS}")
    print(f"  dtheta_dS_pc = {dtheta_dS_pc_correct:.10f}  (units: $/day, correct!)")
    print()
    
    print("Impact on residual:")
    overestimate = dtheta_dS_pc_wrong - dtheta_dS_pc_correct
    print(f"  Before fix was overestimating by: {overestimate:.8f}$/day")
    print(f"  This would inflate residual by ~{abs(overestimate):.2f} per day per contract")
    print()
    
    print("Consistency check with theta handling:")
    theta_prev_year = black_scholes_greeks(S, K, T, r, q, sigma, option_type)[3]
    theta_prev = theta_prev_year / 365.0
    theta_pc = theta_prev * dt
    print(f"  theta_year: {theta_prev_year:.6f}")
    print(f"  theta_daily: {theta_prev:.8f} (theta_year / 365)")
    print(f"  theta_pc: {theta_pc:.8f} (theta_daily × {dt} day)")
    print(f"  ✓ dtheta/dS now follows same daily scaling pattern as theta")
    print()


if __name__ == '__main__':
    compare_scaling()
