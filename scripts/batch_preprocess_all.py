#!/usr/bin/env python
"""
Batch preprocessing: Apply vol surface repair to all market data CSVs.
"""

import os
import sys
import subprocess
from pathlib import Path

def batch_preprocess(backtests_dir='backtests'):
    """Preprocess all market data CSVs in backtests folder."""
    
    backtests_path = Path(backtests_dir)
    if not backtests_path.exists():
        print(f"ERROR: {backtests_dir} not found")
        sys.exit(1)
    
    # Find all mkt_data.csv files
    mkt_files = sorted(backtests_path.glob('*/mkt_data.csv'))
    
    if not mkt_files:
        print(f"No market data files found in {backtests_dir}")
        return
    
    print("="*100)
    print(f"BATCH PREPROCESSING: {len(mkt_files)} datasets")
    print("="*100)
    print("Repairing ITM prices using OTM vol surface...\n")
    
    for i, mkt_file in enumerate(mkt_files, 1):
        subfolder = mkt_file.parent.name
        output_file = mkt_file.parent / "clean_mkt_data.csv"
        
        print(f"[{i}/{len(mkt_files)}] {subfolder}...")
        
        # Run preprocessing
        cmd = ['python', 'scripts/preprocess_vol_repair.py', str(mkt_file), str(output_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            continue
        
        # Extract summary
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Repaired' in line or 'Output:' in line:
                print(f"  {line}")
    
    print("\n" + "="*100)
    print("BATCH COMPLETE")
    print("All datasets now have clean_mkt_data.csv ready for backtesting")
    print("\nUsage:")
    print("  python scripts/fit_and_evolve.py <subfolder> --market-csv backtests/<subfolder>/clean_mkt_data.csv")
    print("  OR just run normally (preprocessing is automatic)")
    print("="*100)

if __name__ == '__main__':
    batch_preprocess()
