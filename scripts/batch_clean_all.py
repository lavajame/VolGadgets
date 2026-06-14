#!/usr/bin/env python
"""
Batch validate and clean all market data in backtests folder.
"""

import os
import sys
import subprocess
from pathlib import Path

def batch_validate_and_clean(backtests_dir='backtests', strategy='moderate'):
    """
    Validate and clean all market data CSVs in backtests folder.
    """
    backtests_path = Path(backtests_dir)
    
    if not backtests_path.exists():
        print(f"ERROR: {backtests_dir} not found")
        sys.exit(1)
    
    # Find all mkt_data.csv files
    mkt_files = list(backtests_path.glob('*/mkt_data.csv'))
    mkt_files.sort()
    
    if not mkt_files:
        print(f"No market data files found in {backtests_dir}")
        return
    
    print("="*100)
    print(f"BATCH VALIDATION & CLEANING: {len(mkt_files)} datasets")
    print(f"Strategy: {strategy}")
    print("="*100)
    
    for i, mkt_file in enumerate(mkt_files, 1):
        subfolder = mkt_file.parent.name
        output_file = mkt_file.parent / f"mkt_data_{strategy}.csv"
        
        print(f"\n[{i}/{len(mkt_files)}] {subfolder}")
        print("-" * 100)
        
        # Validate
        print(f"  Validating: {mkt_file.name}")
        validate_cmd = ['python', 'scripts/validate_mkt_data.py', str(mkt_file)]
        result = subprocess.run(validate_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            continue
        
        # Show key stats
        lines = result.stdout.split('\n')
        for line in lines:
            if 'CRITICAL' in line or 'sudden drops' in line or 'violations' in line:
                print(f"  {line}")
        
        # Clean
        print(f"  Cleaning with {strategy} strategy...")
        clean_cmd = ['python', 'scripts/clean_mkt_data.py', str(mkt_file), str(output_file), strategy]
        result = subprocess.run(clean_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            continue
        
        # Extract final line
        for line in result.stdout.split('\n'):
            if 'Output:' in line:
                print(f"  {line}")
                break
    
    print("\n" + "="*100)
    print("BATCH COMPLETE")
    print(f"Cleaned data saved as mkt_data_{strategy}.csv in each subfolder")
    print("="*100)

if __name__ == '__main__':
    strategy = sys.argv[1] if len(sys.argv) > 1 else 'moderate'
    batch_validate_and_clean(strategy=strategy)
