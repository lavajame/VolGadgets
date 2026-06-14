#!/usr/bin/env python
"""
Market data cleaning and filtering script.
Provides multiple strategies for handling data quality issues.
"""

import pandas as pd
import numpy as np
import sys

def black_scholes_intrinsic(S, K, opt_type):
    """Calculate intrinsic value."""
    if opt_type.lower() == 'call':
        return max(S - K, 0)
    else:
        return max(K - S, 0)

def clean_mkt_data(csv_path, output_path, strategy='conservative'):
    """
    Clean market data according to selected strategy.
    
    Strategies:
    - 'conservative': Remove all suspect records (most aggressive filtering)
    - 'moderate': Remove critical issues, forward-fill zero mids
    - 'lenient': Only remove intrinsic violations
    """
    
    df = pd.read_csv(csv_path, dtype={'contract': str})
    df['quote_date'] = pd.to_datetime(df['quote_date'])
    
    print("="*100)
    print(f"MARKET DATA CLEANING: {strategy.upper()} STRATEGY")
    print("="*100)
    print(f"Input: {len(df)} records")
    
    original_len = len(df)
    
    # Always remove underlying records (they clutter the data)
    df_underlying = df[df['selection_group'] == 'underlying']
    df = df[df['selection_group'] != 'underlying']
    print(f"Removed underlying: -{len(df_underlying)}")
    
    # Filter 1: Intrinsic value violations (always remove - corrupted data)
    options_mask = df['selection_group'].isin(['near_grid', 'far_target']) & df['strike'].notna()
    options_df = df[options_mask].copy()
    
    options_df['intrinsic'] = options_df.apply(
        lambda row: black_scholes_intrinsic(row['underlying_close'], row['strike'], row['option_type']),
        axis=1
    )
    
    intrinsic_violations = options_df[options_df['mid'] < options_df['intrinsic']].copy()
    df = df.drop(intrinsic_violations.index)
    print(f"Removed intrinsic violations: -{len(intrinsic_violations)}")
    
    if strategy in ['conservative', 'moderate']:
        # Filter 2: Bid = 0 anomalies (suspect, likely data errors)
        bid_zero = df[(df['bid'] == 0) & (df['ask'] > 0)]
        df = df.drop(bid_zero.index)
        print(f"Removed bid=0 anomalies: -{len(bid_zero)}")
        
        # Filter 3: Extreme spreads (>50%)
        valid_ba = df[(df['bid'] > 0) & (df['ask'] > 0)].copy()
        valid_ba['spread_pct'] = 100.0 * (valid_ba['ask'] - valid_ba['bid']) / valid_ba['mid']
        extreme_spreads = valid_ba[valid_ba['spread_pct'] > 50]
        df = df.drop(extreme_spreads.index)
        print(f"Removed extreme spreads (>50%): -{len(extreme_spreads)}")
    
    if strategy == 'conservative':
        # Filter 4: Zero mid (forward-fill not allowed in conservative)
        zero_mid = df[df['mid'] == 0]
        df = df.drop(zero_mid.index)
        print(f"Removed zero mid: -{len(zero_mid)}")
    
    elif strategy == 'moderate':
        # Handle zero mid by forward-filling
        # Group by contract and date, forward-fill within each contract
        df_sorted = df.sort_values(['contract', 'quote_date'])
        
        # Mark zero-mid records
        zero_mid_mask = df_sorted['mid'] == 0
        zero_mid_count = zero_mid_mask.sum()
        
        if zero_mid_count > 0:
            # Forward-fill within each contract
            df_sorted['mid_orig'] = df_sorted['mid']
            df_sorted['mid'] = df_sorted.groupby('contract')['mid'].transform(
                lambda x: x.replace(0, np.nan).fillna(method='ffill')
            )
            
            # Count how many were actually filled
            filled = (df_sorted['mid_orig'] == 0) & (df_sorted['mid'].notna())
            filled_count = filled.sum()
            unfilled = (df_sorted['mid_orig'] == 0) & (df_sorted['mid'].isna())
            
            print(f"Zero mid records: {zero_mid_count}")
            print(f"  Forward-filled: {filled_count}")
            print(f"  Could not fill: {len(df_sorted[unfilled])}")
            
            # Remove records that couldn't be filled
            df_sorted = df_sorted.drop(unfilled.index)
            df = df_sorted.drop('mid_orig', axis=1)
    
    # Remove the helper column if it exists
    if 'intrinsic' in df.columns:
        df = df.drop('intrinsic', axis=1)
    
    print(f"\nOutput: {len(df)} records (removed {original_len - len(df)}, {100.0*(original_len-len(df))/original_len:.1f}% reduction)")
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Summary by date
    print(f"\n" + "="*100)
    print("RECORDS PER DATE")
    print("="*100)
    by_date = df.groupby('quote_date').size()
    print(by_date)
    
    return df

def compare_strategies(csv_path):
    """Compare what gets filtered under each strategy."""
    
    df = pd.read_csv(csv_path, dtype={'contract': str})
    df['quote_date'] = pd.to_datetime(df['quote_date'])
    
    strategies = {
        'conservative': "Remove all suspect records",
        'moderate': "Remove critical + forward-fill zeros",
        'lenient': "Only remove intrinsic violations"
    }
    
    print("\n" + "="*100)
    print("STRATEGY COMPARISON")
    print("="*100)
    
    for strategy_name, description in strategies.items():
        print(f"\n{strategy_name.upper()}: {description}")
        print("-" * 100)
        
        # This would call clean_mkt_data with different strategies
        # For now, just show what each would do
        if strategy_name == 'conservative':
            msg = "Remove: intrinsic violations, bid=0, extreme spreads (>50%), zero mid"
        elif strategy_name == 'moderate':
            msg = "Remove: intrinsic violations, bid=0, extreme spreads (>50%)\nForward-fill: zero mid"
        else:
            msg = "Remove: intrinsic violations only"
        
        print(msg)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python clean_mkt_data.py <input.csv> [output.csv] [strategy]")
        print()
        print("Strategies:")
        print("  conservative: Remove all suspect records (most aggressive)")
        print("  moderate:     Remove critical issues, forward-fill zeros (recommended)")
        print("  lenient:      Only remove corrupted records (least aggressive)")
        print()
        print("Example:")
        print("  python clean_mkt_data.py mkt_data.csv mkt_data_clean.csv moderate")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else input_csv.replace('.csv', '_clean.csv')
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'moderate'
    
    if strategy not in ['conservative', 'moderate', 'lenient']:
        print(f"ERROR: Unknown strategy '{strategy}'")
        sys.exit(1)
    
    clean_mkt_data(input_csv, output_csv, strategy=strategy)
