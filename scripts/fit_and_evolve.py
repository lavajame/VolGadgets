#!/usr/bin/env python
"""
Unified script: fit initial portfolio and evolve it over time.
Runs solver and plotter in sequence with a single command.
"""

import argparse
import subprocess
import sys


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with exit code {result.returncode}")
        return False
    return True


def main():
    p = argparse.ArgumentParser(
        description='Fit portfolio and evolve it over time in one command'
    )
    p.add_argument('subfolder', help="Backtests subfolder (e.g., 130206_SPX)")
    p.add_argument('--alpha', type=float, default=1e-4, help='Ridge regularization intensity')
    p.add_argument('--target-type', choices=['call', 'put'], default='call', help='Far target option type')
    p.add_argument('--constraints', type=str, default='delta;gamma', help='Semicolon-delimited list of constraints (e.g., "delta;gamma;vanna;theta"). Always includes mid.')
    p.add_argument('--S-min', type=float, help='Lower spot boundary for asymptotic intrinsic value matching (default: 0)')
    p.add_argument('--S-max', type=float, help='Upper spot boundary for asymptotic intrinsic value matching (default: 10x initial spot)')
    p.add_argument('--market-csv', help='Custom market data CSV (default: backtests/<subfolder>/mkt_data.csv)')
    p.add_argument('--no-preprocess', action='store_true', help='Skip vol surface repair preprocessing')
    p.add_argument('--no-delta-hedge', action='store_true', help='Disable daily delta hedge (enabled by default)')
    args = p.parse_args()

    print(f"\n{'#'*70}")
    print(f"# FIT AND EVOLVE: {args.subfolder}")
    print(f"# Alpha: {args.alpha}")
    print(f"# Constraints: {args.constraints}")
    if args.S_min is not None and args.S_max is not None:
        print(f"# Asymptotic bounds: S_min={args.S_min:.0f}, S_max={args.S_max:.0f}")
    else:
        print(f"# Asymptotic bounds: auto (S_min=0, S_max=10*S0)")
    if not args.no_delta_hedge:
        print(f"# Delta hedge: daily rebalanced")
    else:
        print(f"# Delta hedge: disabled")
    print(f"{'#'*70}")

    # Determine market data file
    if args.market_csv:
        market_data_file = args.market_csv
    else:
        market_data_file = f"backtests/{args.subfolder}/mkt_data.csv"
    
    # Pre-process: vol surface repair if not disabled
    if not args.no_preprocess:
        print("\n>>> STEP -1: PRE-PROCESSING MARKET DATA (Vol Surface Repair)")
        preprocessed_file = market_data_file.replace('.csv', '_preprocessed.csv')
        cmd_preprocess = [
            'python', 'scripts/preprocess_vol_repair.py', market_data_file, preprocessed_file
        ]
        if not run_command(cmd_preprocess, "Preprocessing: repairing ITM prices from OTM vols"):
            sys.exit(1)
        market_data_file = preprocessed_file
        print(f"   → Using preprocessed data: {market_data_file}")
    
    # Step 0: Always pre-compute second-order greeks
    print("\n>>> STEP 0: PRE-COMPUTING SECOND-ORDER GREEKS")
    cmd_so_greeks = [
        'python', 'scripts/compute_second_order_greeks.py', args.subfolder,
        '--market-csv', market_data_file
    ]
    if not run_command(cmd_so_greeks, "Computing vanna, volga, dtheta/dS"):
        sys.exit(1)

    # Step 1: Solve for weights
    print("\n>>> STEP 1: SOLVING FOR NEAR PORTFOLIO WEIGHTS")
    cmd_solver = [
        'python', 'scripts/scale_invariant_ridge_solver.py', args.subfolder,
        '--alpha', str(args.alpha),
        '--constraints', args.constraints,
        '--target-type', args.target_type,
        '--market-csv', market_data_file
    ]
    if args.S_min is not None and args.S_max is not None:
        cmd_solver.extend(['--S-min', str(args.S_min), '--S-max', str(args.S_max)])
    if not args.no_delta_hedge:
        cmd_solver.append('--delta-hedge')
    
    if not run_command(cmd_solver, "Solver: fitting near portfolio"):
        sys.exit(1)

    # Step 2: Plot evolution
    print("\n>>> STEP 2: PLOTTING EVOLUTION")
    cmd_plotter = [
        'python', 'scripts/plot_evolution.py', args.subfolder,
        '--target-type', args.target_type
    ]
    if not args.no_delta_hedge:
        cmd_plotter.append('--delta-hedge')
    
    if not run_command(cmd_plotter, "Plotter: generating evolution plots and table"):
        sys.exit(1)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUCCESS")
    print(f"{'='*70}")
    print(f"Results in: backtests/{args.subfolder}/")
    print(f"  - near_weights.csv: portfolio weights")
    print(f"  - evolution.csv: Greeks evolution over time")
    print(f"  - mid_evolution.png: price evolution")
    print(f"  - greeks_evolution.png: Greeks over time")
    print(f"  - near_weights_by_strike.png: weight distribution by strike")
    print(f"  - pnl_attribution_cumulative_*.png: PnL decomposition")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
