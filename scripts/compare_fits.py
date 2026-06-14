#!/usr/bin/env python
"""
Compare first-order and second-order greeks fitting with delta hedge.
Runs solver and plots for both, backing up first-order results before second-order.
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with exit code {result.returncode}")
        return False
    return True


def main():
    p = argparse.ArgumentParser(
        description='Compare first-order vs second-order greeks fitting with delta hedge'
    )
    p.add_argument('subfolder', help="Backtests subfolder (e.g., 130206_SPX)")
    p.add_argument('--alpha', type=float, default=1e-4, help='Ridge regularization intensity')
    p.add_argument('--target-type', choices=['call', 'put'], default='call', help='Far target option type')
    args = p.parse_args()

    base_dir = os.path.join('backtests', args.subfolder)
    backup_dir = os.path.join('backtests', f'{args.subfolder}_firstorder')

    print(f"\n{'#'*70}")
    print(f"# COMPARING FIRST-ORDER VS SECOND-ORDER GREEKS FITTING")
    print(f"# Subfolder: {args.subfolder}")
    print(f"# Alpha: {args.alpha}")
    print(f"{'#'*70}\n")

    # Step 1: First-order fit with delta hedge
    print("\n>>> STEP 1: FIRST-ORDER FIT (mid, delta, gamma only)")
    cmd1_solver = [
        'python', 'scripts/scale_invariant_ridge_solver.py', args.subfolder,
        '--delta-hedge', '--alpha', str(args.alpha), '--target-type', args.target_type
    ]
    if not run_command(cmd1_solver, "First-order solver"):
        return

    cmd1_plot = [
        'python', 'scripts/plot_evolution.py', args.subfolder,
        '--delta-hedge', '--target-type', args.target_type
    ]
    if not run_command(cmd1_plot, "First-order plots"):
        return

    # Step 2: Backup first-order results
    print(f"\n>>> STEP 2: BACKING UP FIRST-ORDER RESULTS to {backup_dir}")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
        print(f"Removed existing backup directory: {backup_dir}")
    
    shutil.copytree(base_dir, backup_dir)
    print(f"Backed up {base_dir} -> {backup_dir}")

    # Step 3: Second-order fit with delta hedge
    print("\n>>> STEP 3: SECOND-ORDER FIT (including vanna, volga, dtheta/dS)")
    cmd2_solver = [
        'python', 'scripts/scale_invariant_ridge_solver.py', args.subfolder,
        '--include-second-order', '--delta-hedge', '--alpha', str(args.alpha), '--target-type', args.target_type
    ]
    if not run_command(cmd2_solver, "Second-order solver"):
        return

    cmd2_plot = [
        'python', 'scripts/plot_evolution.py', args.subfolder,
        '--delta-hedge', '--target-type', args.target_type
    ]
    if not run_command(cmd2_plot, "Second-order plots"):
        return

    # Summary
    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"First-order results:  {backup_dir}")
    print(f"Second-order results: {base_dir}")
    print(f"\nTo compare the results:")
    print(f"  - Check near_weights.csv in both directories for weight differences")
    print(f"  - Compare plots (mid_evolution.png, greeks_evolution.png, etc.)")
    print(f"  - Review near_weights_by_strike.png to see weight distribution")
    print(f"  - Check PnL attribution charts (pnl_attribution_cumulative_*.png)")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
