#!/usr/bin/env python
"""
Market data validation and quality report.
Identifies data quality issues in mkt_data.csv that could affect backtest results.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def black_scholes_intrinsic(S, K, opt_type):
    """Calculate intrinsic value."""
    if opt_type.lower() == 'call':
        return max(S - K, 0)
    else:
        return max(K - S, 0)

def validate_mkt_data(csv_path):
    """Comprehensive data validation."""
    
    df = pd.read_csv(csv_path, dtype={'contract': str})
    df['quote_date'] = pd.to_datetime(df['quote_date'])
    
    print("="*100)
    print("MARKET DATA VALIDATION REPORT")
    print("="*100)
    print(f"File: {csv_path}")
    print(f"Records: {len(df)}")
    print(f"Date range: {df['quote_date'].min().date()} to {df['quote_date'].max().date()}")
    print(f"Unique dates: {df['quote_date'].nunique()}")
    print()
    
    issues = {}
    
    # 1. CHECK: Missing data
    print("="*100)
    print("1. MISSING DATA CHECK")
    print("="*100)
    
    nan_cols = df.isnull().sum()
    if nan_cols.sum() > 0:
        print("\nColumns with NaN values:")
        for col, count in nan_cols[nan_cols > 0].items():
            pct = 100.0 * count / len(df)
            print(f"  {col}: {count} ({pct:.2f}%)")
        issues['missing_data'] = nan_cols[nan_cols > 0].to_dict()
    else:
        print("✓ No NaN values found")
    
    # 2. CHECK: Quote missing flags
    print("\n" + "="*100)
    print("2. QUOTE MISSING FLAGS")
    print("="*100)
    
    quote_missing_count = (df['quote_missing'] == 1).sum()
    if quote_missing_count > 0:
        print(f"\n⚠ {quote_missing_count} records marked as quote_missing=1 ({100.0*quote_missing_count/len(df):.2f}%)")
        by_date = df[df['quote_missing'] == 1].groupby('quote_date').size()
        print(f"  Affected dates: {len(by_date)}")
        print(f"  Worst date: {by_date.idxmax().date()} with {by_date.max()} missing")
        issues['quote_missing'] = quote_missing_count
    else:
        print("✓ No quote_missing flags")
    
    # 3. CHECK: Zero/negative prices
    print("\n" + "="*100)
    print("3. ZERO/NEGATIVE PRICES")
    print("="*100)
    
    zero_bid = (df['bid'] == 0).sum()
    zero_ask = (df['ask'] == 0).sum()
    zero_mid = (df['mid'] == 0).sum()
    neg_mid = (df['mid'] < 0).sum()
    
    if zero_bid > 0 or zero_ask > 0 or zero_mid > 0 or neg_mid > 0:
        print(f"\n⚠ Price anomalies found:")
        print(f"  Bid = 0: {zero_bid}")
        print(f"  Ask = 0: {zero_ask}")
        print(f"  Mid = 0: {zero_mid}")
        print(f"  Mid < 0: {neg_mid}")
        
        # Show examples
        zero_mid_df = df[(df['mid'] == 0) & (df['selection_group'] == 'near_grid')]
        if len(zero_mid_df) > 0:
            print(f"\n  Example zero-mid near_grid contracts:")
            sample = zero_mid_df[['quote_date', 'contract', 'mid', 'bid', 'ask', 'quote_missing']].drop_duplicates().head(5)
            for _, row in sample.iterrows():
                print(f"    {row['quote_date'].date()}: {row['contract']} mid={row['mid']}, bid={row['bid']}, ask={row['ask']}, missing={row['quote_missing']}")
        
        issues['zero_prices'] = {'zero_mid': zero_mid, 'zero_bid': zero_bid, 'zero_ask': zero_ask, 'neg_mid': neg_mid}
    else:
        print("✓ No zero or negative prices")
    
    # 4. CHECK: Bid-ask spreads
    print("\n" + "="*100)
    print("4. BID-ASK SPREAD ANALYSIS")
    print("="*100)
    
    # Only look at records with valid bid/ask
    valid_ba = df[(df['bid'] > 0) & (df['ask'] > 0)].copy()
    
    if len(valid_ba) > 0:
        valid_ba['spread'] = valid_ba['ask'] - valid_ba['bid']
        valid_ba['spread_pct'] = 100.0 * valid_ba['spread'] / valid_ba['mid']
        
        extreme_spreads = valid_ba[valid_ba['spread_pct'] > 50]
        if len(extreme_spreads) > 0:
            print(f"\n⚠ {len(extreme_spreads)} records with spread > 50% of mid ({100.0*len(extreme_spreads)/len(valid_ba):.2f}%)")
            print(f"  Max spread %: {valid_ba['spread_pct'].max():.2f}%")
            print(f"  Example:")
            sample = extreme_spreads[['quote_date', 'contract', 'bid', 'ask', 'mid', 'spread_pct']].head(3)
            for _, row in sample.iterrows():
                print(f"    {row['quote_date'].date()}: {row['contract']} bid={row['bid']:.2f}, ask={row['ask']:.2f}, mid={row['mid']:.2f}, spread%={row['spread_pct']:.1f}%")
            issues['extreme_spreads'] = len(extreme_spreads)
        else:
            print("✓ All spreads reasonable (< 50% of mid)")
    
    # 5. CHECK: Intrinsic value violations
    print("\n" + "="*100)
    print("5. INTRINSIC VALUE VIOLATIONS")
    print("="*100)
    
    # Only check options (not underlying)
    options_df = df[df['selection_group'].isin(['near_grid', 'far_target'])].copy()
    options_df = options_df[options_df['strike'].notna()].copy()
    
    options_df['intrinsic'] = options_df.apply(
        lambda row: black_scholes_intrinsic(row['underlying_close'], row['strike'], row['option_type']),
        axis=1
    )
    
    # Mid < intrinsic is a violation (should never happen)
    violations = options_df[options_df['mid'] < options_df['intrinsic']].copy()
    
    if len(violations) > 0:
        print(f"\n❌ CRITICAL: {len(violations)} records with mid < intrinsic ({100.0*len(violations)/len(options_df):.2f}%)")
        print(f"  Example violations:")
        sample = violations[['quote_date', 'contract', 'strike', 'option_type', 'underlying_close', 'mid', 'intrinsic']].head(5)
        for _, row in sample.iterrows():
            print(f"    {row['quote_date'].date()}: {row['contract']} S={row['underlying_close']:.2f} K={row['strike']:.0f} {row['option_type'].upper()} mid={row['mid']:.2f} intrinsic={row['intrinsic']:.2f}")
        issues['intrinsic_violations'] = len(violations)
    else:
        print("✓ No intrinsic value violations")
    
    # 6. CHECK: Sudden disappearances (contract trades then stops)
    print("\n" + "="*100)
    print("6. CONTRACT CONTINUITY CHECK")
    print("="*100)
    
    near_grid = df[df['selection_group'] == 'near_grid'].copy()
    
    disappearances = []
    for contract in near_grid['contract'].unique():
        contract_dates = sorted(near_grid[near_grid['contract'] == contract]['quote_date'].unique())
        
        if len(contract_dates) > 1:
            # Check for gaps (not including final expiration date)
            for i in range(len(contract_dates) - 1):
                d1 = contract_dates[i]
                d2 = contract_dates[i+1]
                day_gap = (d2 - d1).days
                
                if day_gap > 1:  # Gap of more than 1 day
                    # Check if it's not an expiration
                    exp_date = near_grid[near_grid['contract'] == contract]['expiration'].iloc[0]
                    if pd.notna(exp_date):
                        exp_date = pd.to_datetime(exp_date)
                        # If contract reappears after a gap (before expiry), it's suspicious
                        if d2 < exp_date:
                            disappearances.append({
                                'contract': contract,
                                'last_date': d1,
                                'next_date': d2,
                                'gap_days': day_gap,
                                'expiration': exp_date
                            })
    
    if disappearances:
        print(f"\n⚠ {len(disappearances)} contracts with trading gaps:")
        for item in disappearances[:10]:  # Show first 10
            print(f"  {item['contract']} missing {item['gap_days']} days ({item['last_date'].date()} → {item['next_date'].date()})")
        issues['trading_gaps'] = len(disappearances)
    else:
        print("✓ No suspicious trading gaps")
    
    # 7. CHECK: Sudden price drops to zero (the Jan 18 issue)
    print("\n" + "="*100)
    print("7. SUDDEN ZERO-MID EVENTS")
    print("="*100)
    
    near_grid_sorted = near_grid.sort_values(['contract', 'quote_date'])
    
    zero_events = []
    for contract in near_grid['contract'].unique():
        contract_data = near_grid_sorted[near_grid_sorted['contract'] == contract].copy()
        contract_data = contract_data.sort_values('quote_date')
        
        for i in range(len(contract_data) - 1):
            prev_row = contract_data.iloc[i]
            curr_row = contract_data.iloc[i+1]
            
            # Check if mid was non-zero, then became zero
            if prev_row['mid'] > 0 and curr_row['mid'] == 0:
                zero_events.append({
                    'contract': contract,
                    'date_before': prev_row['quote_date'],
                    'date_after': curr_row['quote_date'],
                    'mid_before': prev_row['mid'],
                    'mid_after': curr_row['mid'],
                    'strike': prev_row['strike'],
                    'expiration': prev_row['expiration']
                })
    
    if zero_events:
        print(f"\n❌ {len(zero_events)} sudden drops to mid=0:")
        for event in zero_events[:5]:  # Show first 5
            days_to_exp = (pd.to_datetime(event['expiration']) - event['date_after']).days
            print(f"  {event['contract']} K={event['strike']:.0f}")
            print(f"    {event['date_before'].date()}: mid={event['mid_before']:.2f}")
            print(f"    {event['date_after'].date()}: mid={event['mid_after']:.2f} ({days_to_exp} days to expiry)")
        issues['sudden_zeros'] = len(zero_events)
    else:
        print("✓ No sudden drops to zero mid")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*100)
    
    if not issues:
        print("\n✓ Data looks clean!")
    else:
        print(f"\n⚠ Found {len(issues)} categories of issues")
        print("\nRecommendations:")
        
        if 'sudden_zeros' in issues:
            print(f"\n1. Sudden zero-mid events ({issues['sudden_zeros']} found):")
            print("   - OPTION A: Filter out zero-mid records before computing portfolio values")
            print("   - OPTION B: Use bid/ask prices when mid=0")
            print("   - OPTION C: Forward-fill previous valid mid for continuity")
        
        if 'intrinsic_violations' in issues:
            print(f"\n2. Intrinsic violations ({issues['intrinsic_violations']} found):")
            print("   - CRITICAL: These records are corrupted, should be excluded")
        
        if 'extreme_spreads' in issues:
            print(f"\n3. Extreme spreads ({issues['extreme_spreads']} found):")
            print("   - Consider filtering records with spread > 50% of mid as unreliable")
        
        if 'quote_missing' in issues:
            print(f"\n4. Missing quotes ({issues['quote_missing']} found):")
            print("   - Already flagged in data; consider excluding when quote_missing=1")
    
    return issues


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_mkt_data.py <mkt_data.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    validate_mkt_data(csv_path)
