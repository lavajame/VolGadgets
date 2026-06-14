#!/usr/bin/env python
"""
Test dtheta/dS computation against Black-Scholes for scaling issues.
"""

import numpy as np
from bs_pricer import black_scholes_price, black_scholes_greeks


def compute_dtheta_dS_finite_diff(S, K, T, r, q, sigma, option_type, eps_S_frac=1e-3):
    """Compute dtheta/dS via finite differences (per-year theta derivative)."""
    eps_S = eps_S_frac * S
    
    theta_up = black_scholes_greeks(S + eps_S, K, T, r, q, sigma, option_type)[3]
    theta_dn = black_scholes_greeks(S - eps_S, K, T, r, q, sigma, option_type)[3]
    
    dtheta_dS_year = (theta_up - theta_dn) / (2 * eps_S)
    return dtheta_dS_year


def test_dtheta_dS():
    """Test dtheta/dS calculation."""
    S = 1462.42
    K = 1490.0
    T = 0.157534  # ~57 days
    r = 0.0
    q = 0.0
    sigma = 0.0288
    option_type = 'call'
    
    print("Testing dtheta/dS computation")
    print("=" * 70)
    print(f"S={S}, K={K}, T={T:.6f} years (~{T*365:.0f} days), sigma={sigma:.4f}")
    print()
    
    # Compute base greeks at S
    delta, gamma, vega, theta_year = black_scholes_greeks(S, K, T, r, q, sigma, option_type)
    theta_day = theta_year / 365.0
    
    print(f"Greeks at S={S}:")
    print(f"  delta: {delta:.6f}")
    print(f"  gamma: {gamma:.8f}")
    print(f"  vega: {vega:.6f}")
    print(f"  theta (per-year): {theta_year:.6f}")
    print(f"  theta (per-day): {theta_day:.8f}")
    print()
    
    # Compute dtheta/dS in different ways
    dtheta_dS_year = compute_dtheta_dS_finite_diff(S, K, T, r, q, sigma, option_type)
    dtheta_dS_day = dtheta_dS_year / 365.0
    
    print(f"dtheta/dS (per-year theta / per-unit S): {dtheta_dS_year:.8f}")
    print(f"dtheta/dS (per-day theta / per-unit S):  {dtheta_dS_day:.10f}")
    print()
    
    # Test: what is dtheta/dS × dS?
    dS_values = [1.0, 5.0, 10.0]
    print("PnL impact of dtheta/dS × dS (using per-year derivative):")
    for dS in dS_values:
        # Using per-year derivative
        pnl_year = dtheta_dS_year * dS
        # Using per-day derivative
        pnl_day = dtheta_dS_day * dS
        print(f"  dS={dS:5.1f}: year-based={pnl_year:10.6f}, day-based={pnl_day:12.8f}")
    print()
    
    # Sanity check: compare analytical vs finite difference
    # dtheta/dS ≈ d²price/dS/dt (cross partial derivative)
    # Using Black-Scholes formula properties
    print("Sanity check: theta × gamma relationship")
    print(f"  gamma: {gamma:.8f}")
    print(f"  dtheta/dS / gamma: {dtheta_dS_year / gamma:.6f}")
    print(f"  (dtheta/dS is typically opposite sign to gamma * S * vega)")
    print()
    
    # Test with perturbations
    print("Verification: second-order finite difference check")
    eps = 0.001
    price_center = black_scholes_price(S, K, T, r, q, sigma, option_type)
    price_up = black_scholes_price(S + eps, K, T, r, q, sigma, option_type)
    price_dn = black_scholes_price(S - eps, K, T, r, q, sigma, option_type)
    
    delta_fd = (price_up - price_dn) / (2 * eps)
    gamma_fd = (price_up - 2*price_center + price_dn) / eps**2
    
    print(f"  delta (from price FD): {delta_fd:.6f}")
    print(f"  gamma (from price FD): {gamma_fd:.8f}")
    print()
    
    # Check if dtheta/dS makes sense dimensionally
    print("Dimensional analysis:")
    print(f"  theta units: $/year (premium decay per year)")
    print(f"  dtheta/dS units: $/year per $ of spot")
    print(f"  dtheta/dS × dS × (1 day / 365 days) would give daily PnL from dtheta/dS term")
    print(f"  Expected term in daily PnL: {dtheta_dS_year * 1.0 / 365.0:.10f} for 1$ move")
    print()
    
    # What should we use in the attribution?
    print("Recommended usage in PnL attribution:")
    print(f"  Option 1: Use dtheta_dS_year * dS (units: $/year)")
    print(f"           Then divide by 365 when computing daily PnL")
    print(f"           Current implementation: dtheta_dS_year * dS = {dtheta_dS_year * 1.0:.8f} for dS=1")
    print()
    print(f"  Option 2: Use dtheta_dS_day * dS (units: $/day)")
    print(f"           Daily PnL directly without scaling")
    print(f"           Would need: dtheta_dS_day * dS = {dtheta_dS_day * 1.0:.10f} for dS=1")
    print()


if __name__ == '__main__':
    test_dtheta_dS()
