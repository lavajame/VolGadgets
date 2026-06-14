#!/usr/bin/env python
"""
Quick comparison of different constraint configurations for vega isolation.
"""

import subprocess
import os
import sys
import pandas as pd
from pathlib import Path

def run_config(subfolder, constraints, name):
    """Run fit_and_evolve with a specific constraint configuration."""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"Constraints: {constraints}")
    print(f"{'='*70}")
    
    cmd = [
        'python', 'scripts/fit_and_evolve.py', subfolder,
        '--constraints', constraints,
        '--no-delta-hedge'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {name} failed")
        print(result.stderr)
        return None
    
    # Read the evolution and compute vega isolation score
    evo_path = os.path.join('backtests', subfolder, 'evolution.csv')
    if not os.path.exists(evo_path):
        print(f"ERROR: {evo_path} not found")
        return None
    
    evo = pd.read_csv(evo_path)
    row0 = evo[evo['quote_date'] == evo['quote_date'].min()].iloc[0]
    
    # Metrics at day 0
    delta = float(row0['net_delta']) if pd.notna(row0['net_delta']) else 0
    gamma = float(row0['net_gamma']) if pd.notna(row0['net_gamma']) else 0
    theta = float(row0['net_theta']) if pd.notna(row0['net_theta']) else 0
    vega = float(row0['net_vega']) if pd.notna(row0['net_vega']) else 0
    
    return {
        'name': name,
        'constraints': constraints,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }


def main():
    subfolder = '130301_SPX_Wide'
    
    # Baseline: delta, gamma, theta
    # Then add individual second-order greeks to see their impact
    configs = [
        ('delta;gamma;theta', 'Baseline: 1st order only'),
        ('delta;gamma;theta;vanna', 'Baseline + Vanna'),
        ('delta;gamma;theta;volga', 'Baseline + Volga'),
        ('delta;gamma;theta;dtheta_dS_year', 'Baseline + dTheta/dS'),
        ('delta;gamma;theta;vanna;volga', 'Baseline + Vanna + Volga'),
        ('delta;gamma;theta;vanna;dtheta_dS_year', 'Baseline + Vanna + dTheta/dS'),
        ('delta;gamma;theta;volga;dtheta_dS_year', 'Baseline + Volga + dTheta/dS'),
        ('delta;gamma;theta;vanna;volga;dtheta_dS_year', 'All 2nd order'),
    ]
    
    print(f"\n{'#'*70}")
    print(f"# VEGA ISOLATION COMPARISON: {subfolder}")
    print(f"# Baseline: delta;gamma;theta (1st-order only)")
    print(f"# Adding: individual and combined 2nd-order Greeks")
    print(f"# Auto-bounds enabled (S_min=0, S_max=10*S0)")
    print(f"# No delta hedge (natural isolation)")
    print(f"{'#'*70}")
    
    results = []
    for constraints, name in configs:
        r = run_config(subfolder, constraints, name)
        if r:
            results.append(r)
    
    # Print comparison table
    print(f"\n{'='*110}")
    print(f"VEGA ISOLATION QUALITY METRICS (Day 0 Greeks)")
    print(f"{'='*110}")
    
    print(f"\n{'Config':<50} {'Delta':>10} {'Gamma':>12} {'Theta':>10} {'Vega':>10} {'Score':>10}")
    print("-" * 110)
    
    for r in results:
        delta_score = abs(r['delta'])
        gamma_score = abs(r['gamma'])
        theta_score = abs(r['theta'])
        vega_score = abs(r['vega'])
        
        # Quality: how well does this isolate vega?
        # Good if: delta ≈ 0, gamma ≈ 0, theta ≈ 0, vega >> others
        greek_noise = delta_score + gamma_score + abs(theta_score)
        quality = "✓✓✓" if greek_noise < 0.01 else "✓✓" if greek_noise < 0.1 else "✓" if greek_noise < 1.0 else "✗"
        
        print(f"{r['name']:<50} {delta_score:>10.6f} {gamma_score:>12.8f} {theta_score:>10.2f} {vega_score:>10.2f} {quality:>10s}")
    
    print(f"\n{'='*110}")
    print("ANALYSIS:")
    print("- Quality score ✓✓✓ = Excellent isolation (greek noise < 0.01)")
    print("- Quality score ✓✓  = Good isolation (greek noise < 0.1)")
    print("- Quality score ✓   = Fair isolation (greek noise < 1.0)")
    print("- Quality score ✗   = Poor isolation (greek noise > 1.0)")
    print()
    print("Greek noise = |Delta| + |Gamma| + |Theta|")
    print("Goal: Minimize greek noise while maximizing pure Vega exposure")
    print(f"{'='*110}\n")


if __name__ == '__main__':
    main()
